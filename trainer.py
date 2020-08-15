"""
Copyright (C) 2020 Hsin-Yu Chang <acht7111020@gmail.com>
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import os

import torch
import torch.nn as nn
from torch.autograd import Variable

from networks import AdaINGen, MsImageDis, ContentEncoder_share
from utils import weights_init, get_model_list, vgg_preprocess, get_scheduler
from layers import contextual_loss


class DSMAP_Trainer(nn.Module):
    def __init__(self, hyperparameters):
        super(DSMAP_Trainer, self).__init__()
        # Initiate the networks
        mid_downsample = hyperparameters['gen'].get('mid_downsample', 1)
        self.content_enc = ContentEncoder_share(hyperparameters['gen']['n_downsample'],
                                                mid_downsample,
                                                hyperparameters['gen']['n_res'],
                                                hyperparameters['input_dim_a'],
                                                hyperparameters['gen']['dim'],
                                                'in',
                                                hyperparameters['gen']['activ'],
                                                pad_type=hyperparameters['gen']['pad_type'])

        self.style_dim = hyperparameters['gen']['style_dim']
        self.gen_a = AdaINGen(hyperparameters['input_dim_a'], self.content_enc, 'a', hyperparameters['gen'])  # auto-encoder for domain a
        self.gen_b = AdaINGen(hyperparameters['input_dim_b'], self.content_enc, 'b', hyperparameters['gen'])  # auto-encoder for domain b
        self.dis_a = MsImageDis(hyperparameters['input_dim_a'], self.content_enc.output_dim, hyperparameters['dis'])  # discriminator for domain a
        self.dis_b = MsImageDis(hyperparameters['input_dim_b'], self.content_enc.output_dim, hyperparameters['dis'])  # discriminator for domain b

    def build_optimizer(self, hyperparameters):
        # Setup the optimizers
        lr = hyperparameters['lr']
        beta1 = hyperparameters['beta1']
        beta2 = hyperparameters['beta2']
        dis_params = list(self.dis_a.parameters()) + list(self.dis_b.parameters())
        gen_params = list(self.gen_a.parameters()) + list(self.gen_b.parameters())
        self.dis_opt = torch.optim.Adam([p for p in dis_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        self.gen_opt = torch.optim.Adam([p for p in gen_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters)
        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters)

        # Network weight initialization
        self.apply(weights_init(hyperparameters['init']))
        self.dis_a.apply(weights_init('gaussian'))
        self.dis_b.apply(weights_init('gaussian'))

        # Load VGG model if needed
        if 'vgg_w' in hyperparameters.keys() and hyperparameters['vgg_w'] > 0:
            import torchvision.models as models
            self.vgg = models.vgg16(pretrained=True)
            # If you cannot download pretrained model automatically, you can download it from
            # https://download.pytorch.org/models/vgg16-397923af.pth and load it manually
            # state_dict = torch.load('vgg16-397923af.pth')
            # self.vgg.load_state_dict(state_dict)
            self.vgg.eval()
            for param in self.vgg.parameters():
                param.requires_grad = False

    def recon_criterion(self, input, target):
        return torch.mean(torch.abs(input - target))

    def __compute_kl(self, mu, logvar):
        encoding_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return encoding_loss

    def gen_update(self, x_a, x_b, hyperparameters, iterations):
        self.gen_opt.zero_grad()
        self.gen_backward_cc(x_a, x_b, hyperparameters)
        #self.gen_opt.step()

        #self.gen_opt.zero_grad()
        self.gen_backward_latent(x_a, x_b, hyperparameters)
        self.gen_opt.step()

    def gen_backward_latent(self, x_a, x_b, hyperparameters):
        # random sample style vector and multimodal training
        s_a_random = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda())
        s_b_random = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda())
        # decode
        x_ba_random = self.gen_a.decode(self.c_b, self.da_b, s_a_random)
        x_ab_random = self.gen_b.decode(self.c_a, self.db_a, s_b_random)

        c_b_random_recon, _, _, s_a_random_recon, _, _ = self.gen_a.encode(x_ba_random)
        c_a_random_recon, _, _, s_b_random_recon, _, _ = self.gen_b.encode(x_ab_random)

        # style reconstruction loss
        self.loss_gen_recon_s_a = self.recon_criterion(s_a_random, s_a_random_recon)
        self.loss_gen_recon_s_b = self.recon_criterion(s_b_random, s_b_random_recon)
        loss_gen_recon_c_a = self.recon_criterion(self.c_a, c_a_random_recon)
        loss_gen_recon_c_b = self.recon_criterion(self.c_b, c_b_random_recon)
        loss_gen_adv_a = self.dis_a.calc_gen_loss(x_ba_random, x_a)
        loss_gen_adv_b = self.dis_b.calc_gen_loss(x_ab_random, x_b)
        loss_gen_vgg_a = self.compute_vgg_loss(self.vgg, x_ba_random, x_a) if hyperparameters['vgg_w'] > 0 else 0
        loss_gen_vgg_b = self.compute_vgg_loss(self.vgg, x_ab_random, x_b) if hyperparameters['vgg_w'] > 0 else 0

        loss_gen_total = hyperparameters['gan_w'] * loss_gen_adv_a + \
                          hyperparameters['gan_w'] * loss_gen_adv_b + \
                          hyperparameters['recon_s_w'] * self.loss_gen_recon_s_a + \
                          hyperparameters['recon_s_w'] * self.loss_gen_recon_s_b + \
                          hyperparameters['recon_c_w'] * loss_gen_recon_c_a + \
                          hyperparameters['recon_c_w'] * loss_gen_recon_c_b + \
                          hyperparameters['vgg_w'] * loss_gen_vgg_a + \
                          hyperparameters['vgg_w'] * loss_gen_vgg_b

        self.loss_gen_total += loss_gen_total
        self.loss_gen_total.backward()

        self.loss_gen_total += loss_gen_total
        self.loss_gen_adv_a += loss_gen_adv_a
        self.loss_gen_adv_b += loss_gen_adv_b
        self.loss_gen_recon_c_a += loss_gen_recon_c_a
        self.loss_gen_recon_c_b += loss_gen_recon_c_b
        self.loss_gen_vgg_a += loss_gen_vgg_a
        self.loss_gen_vgg_b += loss_gen_vgg_b

    def gen_backward_cc(self, x_a, x_b, hyperparameters):
        pre_c_a, self.c_a, c_domain_a, self.db_a, self.s_a_prime, mu_a, logvar_a = self.gen_a.encode(x_a, training=True, flag=True)
        pre_c_b, self.c_b, c_domain_b, self.da_b, self.s_b_prime, mu_b, logvar_b = self.gen_b.encode(x_b, training=True, flag=True)

        self.da_a = self.gen_b.domain_mapping(self.c_a, pre_c_a)
        self.db_b = self.gen_a.domain_mapping(self.c_b, pre_c_b)

        # decode (within domain)
        x_a_recon = self.gen_a.decode(self.c_a, self.da_a, self.s_a_prime)
        x_b_recon = self.gen_b.decode(self.c_b, self.db_b, self.s_b_prime)
        # decode (cross domain)
        x_ba = self.gen_a.decode(self.c_b, self.da_b, self.s_a_prime)
        x_ab = self.gen_b.decode(self.c_a, self.db_a, self.s_b_prime)

        c_b_recon, _, self.db_b_recon, s_a_recon, _, _ = self.gen_a.encode(x_ba, training=True)
        c_a_recon, _, self.da_a_recon, s_b_recon, _, _ = self.gen_b.encode(x_ab, training=True)
        # decode again (cycle consistance loss)
        x_aba = self.gen_a.decode(c_a_recon, self.da_a_recon, s_a_recon )
        x_bab = self.gen_b.decode(c_b_recon, self.db_b_recon, s_b_recon )

        # domain-specific content reconstruction loss
        self.loss_gen_recon_d_a = self.recon_criterion(c_domain_a, self.da_a) if hyperparameters['recon_d_w'] > 0 else 0
        self.loss_gen_recon_d_b = self.recon_criterion(c_domain_b, self.db_b) if hyperparameters['recon_d_w'] > 0 else 0

        # image reconstruction loss
        self.loss_gen_recon_x_a = self.recon_criterion(x_a_recon, x_a)
        self.loss_gen_recon_x_b = self.recon_criterion(x_b_recon, x_b)
        # domain-invariant content reconstruction loss
        self.loss_gen_recon_c_a = self.recon_criterion(c_a_recon, self.c_a)
        self.loss_gen_recon_c_b = self.recon_criterion(c_b_recon, self.c_b)
        # cyc loss
        self.loss_gen_cycrecon_x_a = self.recon_criterion(x_aba, x_a)
        self.loss_gen_cycrecon_x_b = self.recon_criterion(x_bab, x_b)
        # kl loss (if needed)
        self.loss_gen_recon_kl_a = self.__compute_kl(mu_a, logvar_a) if hyperparameters['recon_kl_w'] > 0 else 0
        self.loss_gen_recon_kl_b = self.__compute_kl(mu_b, logvar_b) if hyperparameters['recon_kl_w'] > 0 else 0
        # GAN loss
        self.loss_gen_adv_a = self.dis_a.calc_gen_loss(x_ba, x_a)
        self.loss_gen_adv_b = self.dis_b.calc_gen_loss(x_ab, x_b)
        # domain-invariant perceptual loss
        self.loss_gen_vgg_a = self.compute_vgg_loss(self.vgg, x_ba, x_a) if hyperparameters['vgg_w'] > 0 else 0
        self.loss_gen_vgg_b = self.compute_vgg_loss(self.vgg, x_ab, x_b) if hyperparameters['vgg_w'] > 0 else 0
        # total loss
        self.loss_gen_total = hyperparameters['gan_w'] * self.loss_gen_adv_a + \
                              hyperparameters['gan_w'] * self.loss_gen_adv_b + \
                              hyperparameters['recon_c_w'] * self.loss_gen_recon_c_a + \
                              hyperparameters['recon_c_w'] * self.loss_gen_recon_c_b + \
                              hyperparameters['recon_x_w'] * self.loss_gen_recon_x_a + \
                              hyperparameters['recon_x_w'] * self.loss_gen_recon_x_b + \
                              hyperparameters['recon_d_w'] * self.loss_gen_recon_d_a + \
                              hyperparameters['recon_d_w'] * self.loss_gen_recon_d_b + \
                              hyperparameters['recon_x_cyc_w'] * self.loss_gen_cycrecon_x_a + \
                              hyperparameters['recon_x_cyc_w'] * self.loss_gen_cycrecon_x_b + \
                              hyperparameters['recon_kl_w'] * self.loss_gen_recon_kl_a + \
                              hyperparameters['recon_kl_w'] * self.loss_gen_recon_kl_b + \
                              hyperparameters['vgg_w'] * self.loss_gen_vgg_a + \
                              hyperparameters['vgg_w'] * self.loss_gen_vgg_b

        # self.loss_gen_total.backward(retain_graph=True)

    def compute_vgg_loss(self, vgg, img, target):
        img_vgg = vgg_preprocess(img)
        target_vgg = vgg_preprocess(target)
        img_fea = vgg.features(img_vgg)
        target_fea = vgg.features(target_vgg)
        return contextual_loss(img_fea, target_fea)

    def dis_update(self, x_a, x_b, hyperparameters, iterations):
        self.dis_opt.zero_grad()
        s_a_random = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda())
        s_b_random = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda())

        # encode
        pre_c_a, c_a, c_domain_a, db_a, s_a, _, _ = self.gen_a.encode(x_a, training=True, flag=True)
        pre_c_b, c_b, c_domain_b, da_b, s_b, _, _ = self.gen_b.encode(x_b, training=True, flag=True)
        da_a = self.gen_b.domain_mapping(c_a, pre_c_a)
        db_b = self.gen_a.domain_mapping(c_b, pre_c_b)
        # decode (cross domain)
        x_ba = self.gen_a.decode(c_b, da_b, s_a)
        x_ab = self.gen_b.decode(c_a, db_a, s_b)
        # decode (cross domain)
        x_ba_random = self.gen_a.decode(c_b, da_b, s_a_random)
        x_ab_random = self.gen_b.decode(c_a, db_a, s_b_random)

        c_b_recon, _, db_b_recon, s_a_recon, _, _ = self.gen_a.encode(x_ba)
        c_a_recon, _, da_a_recon, s_b_recon, _, _ = self.gen_b.encode(x_ab)
        _, _, db_b_random_recon, _, _, _ = self.gen_a.encode(x_ba_random)
        _, _, da_a_random_recon, _, _, _ = self.gen_b.encode(x_ab_random)

        # D loss
        self.loss_dis_a = self.dis_a.calc_dis_loss(x_ba.detach(), x_a) + self.dis_a.calc_dis_loss(x_ba_random.detach(), x_a)
        self.loss_dis_b = self.dis_b.calc_dis_loss(x_ab.detach(), x_b) + self.dis_b.calc_dis_loss(x_ab_random.detach(), x_b)

        self.loss_dis_total = hyperparameters['gan_w'] * self.loss_dis_a + \
                              hyperparameters['gan_w'] * self.loss_dis_b
        self.loss_dis_total.backward()
        self.dis_opt.step()

    def sample(self, x_a, x_b):
        self.eval()
        x_a_recon, x_b_recon, x_ab, x_ba = [], [], [], []
        for i in range(x_a.size(0)):
            pre_c_a, c_a, _, db_a, s_a_fake, _, _ = self.gen_a.encode(x_a[i].unsqueeze(0), flag=True)
            pre_c_b, c_b, _, da_b, s_b_fake, _, _ = self.gen_b.encode(x_b[i].unsqueeze(0), flag=True)
            da_a = self.gen_b.domain_mapping(c_a, pre_c_a)
            db_b = self.gen_a.domain_mapping(c_b, pre_c_b)
            x_a_recon.append(self.gen_a.decode(c_a, da_a, s_a_fake))
            x_b_recon.append(self.gen_b.decode(c_b, db_b, s_b_fake))
            x_ba.append(self.gen_a.decode(c_b, da_b, s_a_fake))
            x_ab.append(self.gen_b.decode(c_a, db_a, s_b_fake))

        x_a_recon, x_b_recon = torch.cat(x_a_recon), torch.cat(x_b_recon)
        x_ab, x_ba = torch.cat(x_ab), torch.cat(x_ba)
        self.train()

        return x_a, x_a_recon, x_ab, x_b, x_b_recon, x_ba

    def update_learning_rate(self):
        if self.dis_scheduler is not None:
            self.dis_scheduler.step()
        if self.gen_scheduler is not None:
            self.gen_scheduler.step()

    def resume(self, checkpoint_dir, hyperparameters):
        # Load generators
        last_model_name = get_model_list(checkpoint_dir, "gen")
        state_dict = torch.load(last_model_name)
        self.gen_a.load_state_dict(state_dict['a'])
        self.gen_b.load_state_dict(state_dict['b'])
        iterations = int(last_model_name[-11:-3])
        # Load discriminators
        last_model_name = get_model_list(checkpoint_dir, "dis")
        state_dict = torch.load(last_model_name)
        self.dis_a.load_state_dict(state_dict['a'])
        self.dis_b.load_state_dict(state_dict['b'])
        # Load optimizers
        state_dict = torch.load(os.path.join(checkpoint_dir, 'optimizer.pt'))
        self.dis_opt.load_state_dict(state_dict['dis'])
        self.gen_opt.load_state_dict(state_dict['gen'])
        # Reinitilize schedulers
        self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters, iterations)
        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters, iterations)
        print('Resume from iteration %d' % iterations)
        return iterations

    def save(self, snapshot_dir, iterations):
        # Save generators, discriminators, and optimizers
        gen_name = os.path.join(snapshot_dir, 'gen_%08d.pt' % (iterations + 1))
        dis_name = os.path.join(snapshot_dir, 'dis_%08d.pt' % (iterations + 1))
        opt_name = os.path.join(snapshot_dir, 'optimizer.pt')
        torch.save({'a': self.gen_a.state_dict(), 'b': self.gen_b.state_dict()}, gen_name)
        torch.save({'a': self.dis_a.state_dict(), 'b': self.dis_b.state_dict()}, dis_name)
        torch.save({'gen': self.gen_opt.state_dict(), 'dis': self.dis_opt.state_dict()}, opt_name)
