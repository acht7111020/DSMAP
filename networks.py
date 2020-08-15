"""
Copyright (C) 2020 Hsin-Yu Chang <acht7111020@gmail.com>
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import torch
from torch import nn
from torch.autograd import Variable

from layers import *


class MsImageDis(nn.Module):
    # Multi-scale discriminator architecture
    def __init__(self, input_dim, content_dim, params):
        super(MsImageDis, self).__init__()
        self.n_layer = params['n_layer']
        self.gan_type = params['gan_type']
        self.dim = params['dim']
        self.norm = params['norm']
        self.activ = params['activ']
        self.num_scales = params['num_scales']
        self.pad_type = params['pad_type']
        self.input_dim = input_dim
        self.content_dim = content_dim
        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)
        self.cnns = nn.ModuleList()
        for i in range(self.num_scales):
            self.cnns.append(self._make_net(i))

    def _make_net(self, idx):
        dim = self.dim
        cnn_x = []
        cnn_x += [Conv2dBlock(self.input_dim, dim, 4, 2, 1, norm='none', activation=self.activ, pad_type=self.pad_type)]
        for i in range(self.n_layer - 1):
            cnn_x += [Conv2dBlock(dim, dim * 2, 4, 2, 1, norm=self.norm, activation=self.activ, pad_type=self.pad_type)]
            dim *= 2
        cnn_x += [nn.Conv2d(dim, 1, 1, 1, 0)]
        cnn_x = nn.Sequential(*cnn_x)
        return cnn_x

    def forward(self, x):
        outputs = []
        for model in self.cnns:
            outputs.append(model(x))
            x = self.downsample(x)
        return outputs

    def calc_dis_loss(self, input_fake, input_real):
        # calculate the loss to train D
        outs0 = self.forward(input_fake)
        outs1 = self.forward(input_real)
        loss = 0

        for it, (out0, out1) in enumerate(zip(outs0, outs1)):
            if self.gan_type == 'lsgan':
                loss += torch.mean((out0 - 0)**2) + torch.mean((out1 - 1)**2)
            elif self.gan_type == 'nsgan':
                all0 = Variable(torch.zeros_like(out0.data).cuda(), requires_grad=False)
                all1 = Variable(torch.ones_like(out1.data).cuda(), requires_grad=False)
                loss += torch.mean(F.binary_cross_entropy(F.sigmoid(out0), all0) +
                                   F.binary_cross_entropy(F.sigmoid(out1), all1))
            elif self.gan_type == 'ralsgan':
                # all1 = Variable(torch.ones_like(out0.data).cuda(), requires_grad=False)
                loss += (torch.mean((out0 - torch.mean(out1) + 1)**2) +
                        torch.mean((out1 - torch.mean(out0) - 1)**2))/2
            else:
                assert 0, "Unsupported GAN type: {}".format(self.gan_type)
        return loss

    def calc_gen_loss(self, input_fake, input_real):
        # calculate the loss to train G
        outs0 = self.forward(input_fake)
        outs1 = self.forward(input_real)

        loss = 0
        for it, (out0, out1) in enumerate(zip(outs0, outs1)):
            if self.gan_type == 'lsgan':
                loss += torch.mean((out0 - 1)**2) # LSGAN
            elif self.gan_type == 'nsgan':
                all1 = Variable(torch.ones_like(out0.data).cuda(), requires_grad=False)
                loss += torch.mean(F.binary_cross_entropy(F.sigmoid(out0), all1))
            elif self.gan_type == 'ralsgan':
                # all1 = Variable(torch.ones_like(out0.data).cuda(), requires_grad=False)
                loss += (torch.mean((out0 - torch.mean(out1) - 1)**2) +
                        torch.mean((out1 - torch.mean(out0) + 1)**2))/2
            else:
                assert 0, "Unsupported GAN type: {}".format(self.gan_type)
        return loss


class AdaINGen(nn.Module):
    # AdaIN auto-encoder architecture
    def __init__(self, input_dim, enc_content, model_type, params):
        super(AdaINGen, self).__init__()
        dim = params['dim']
        style_dim = params['style_dim']
        n_downsample = params['n_downsample']
        n_res = params['n_res']
        activ = params['activ']
        pad_type = params['pad_type']
        mlp_dim = params['mlp_dim']
        mid_downsample = params.get('mid_downsample', 1)
        self.status_d = params.get('status_d', True)
        self.kl_mode = params.get('kl_mode', False)
        self.model_type = model_type

        # style encoder
        if self.kl_mode:
            self.enc_style = StyleEncoder_KL(4, input_dim, dim, style_dim, norm='none', activ='none', pad_type=pad_type)
        else:
            self.enc_style = StyleEncoder(4, input_dim, dim, style_dim, norm='none', activ=activ, pad_type=pad_type)

        # content encoder
        self.enc_content = enc_content
        self.domain_mapping = ResBlock_reverse(self.enc_content.output_dim)

        # decoder
        self.dec = Decoder(n_downsample, mid_downsample, n_res, self.enc_content.output_dim, input_dim, res_norm='adain', activ=activ, pad_type=pad_type)

        # MLP to generate AdaIN parameters
        self.mlp = MLP(style_dim, self.get_num_adain_params(self.dec), mlp_dim, 3, norm='none', activ=activ)

    def encode(self, images, training=False, flag=False):
        # encode an image to its content and style codes
        if self.kl_mode:
            mu, logvar = self.enc_style(images)
            if training == True:
                std = torch.exp(0.5 * logvar)
                eps = torch.randn_like(std)
                style_fake = eps.mul(std).add_(mu)  # z_mean + tf.exp(0.5 * z_var) * eps
            else:
                style_fake = mu

            style_fake = style_fake.unsqueeze(2).unsqueeze(3)
        else:
            style_fake = self.enc_style(images)
            mu, logvar = None, None

        share_content, domain_content, pre_content = self.enc_content(images, self.model_type)
        domain_mapping = self.domain_mapping(share_content, pre_content)

        if flag:
            return pre_content, share_content, domain_content, domain_mapping, style_fake, mu, logvar
        else:
            return share_content, domain_content, domain_mapping, style_fake, mu, logvar

    def decode(self, share_content, domain_mapping, style):
        adain_params = self.mlp(style)
        self.assign_adain_params(adain_params, self.dec)
        if self.status_d:
            images = self.dec(domain_mapping)
        else:
            images = self.dec(share_content)

        return images

    def assign_adain_params(self, adain_params, model):
        # assign the adain_params to the AdaIN layers in model
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                mean = adain_params[:, :m.num_features]
                std = adain_params[:, m.num_features:2*m.num_features]
                m.bias = mean.contiguous().view(-1)
                m.weight = std.contiguous().view(-1)
                if adain_params.size(1) > 2*m.num_features:
                    adain_params = adain_params[:, 2*m.num_features:]

    def get_num_adain_params(self, model):
        # return the number of AdaIN parameters needed by the model
        num_adain_params = 0
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                num_adain_params += 2*m.num_features
        return num_adain_params


##################################################################################
# Encoder and Decoders
##################################################################################
class StyleEncoder(nn.Module):
    def __init__(self, n_downsample, input_dim, dim, style_dim, norm, activ, pad_type):
        super(StyleEncoder, self).__init__()
        self.model = []
        self.model += [Conv2dBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type)]
        for i in range(2):
            self.model += [Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
            dim *= 2
        for i in range(n_downsample - 2):
            self.model += [Conv2dBlock(dim, dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
        self.model += [nn.AdaptiveAvgPool2d(1)] # global average pooling
        self.model += [nn.Conv2d(dim, style_dim, 1, 1, 0)]
        self.model = nn.Sequential(*self.model)
        self.output_dim = dim

    def forward(self, x):
        return self.model(x)


class StyleEncoder_KL(nn.Module):
    def __init__(self, n_layer, input_dim, dim, style_dim, norm, activ, pad_type):
        super(StyleEncoder_KL, self).__init__()
        self.model = []
        self.model += [Conv2dBlock(input_dim, dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]

        for i in range(1, n_layer+1):
            self.model += [BasicBlock(dim*i, dim*(min(i+1, 4)))]
            output_dim = dim*(min(i+1, 4))
#             dim *= 2
        self.model += [nn.LeakyReLU(0.2, inplace=False)]
        self.model += [nn.AdaptiveAvgPool2d(1)] # global average pooling
        self.avgpool = nn.Sequential(*self.model)
        self.mean = nn.Sequential(*[nn.Linear(output_dim, style_dim)])
        self.var = nn.Sequential(*[nn.Linear(output_dim, style_dim)])
        self.output_dim = dim

    def forward(self, x):
        features = self.avgpool(x).view(x.size(0), -1)
        mean = self.mean(features)
        var = self.var(features)
        return mean, var


class ContentEncoder_share(nn.Module):
    def __init__(self, n_downsample, mid_downsample, n_res, input_dim, dim, norm, activ, pad_type):
        super(ContentEncoder_share, self).__init__()
        # Content Encoder A
        self.model = []
        self.model += [Conv2dBlock(input_dim, dim, 7, 1, 3, norm='none', activation='none', pad_type=pad_type)]
        self.model += [nn.LeakyReLU(inplace=True)]
        current_dim = dim

        for i in range(n_downsample):
            self.model += [Conv2dBlock(current_dim, 2 * current_dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
            current_dim *= 2
        self.prelayer_a = nn.Sequential(*self.model)
        self.model = []
        for i in range(mid_downsample):
            self.model += [Conv2dBlock(current_dim, current_dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]

        self.model += [ResBlocks(n_res, current_dim, norm=norm, activation=activ, pad_type=pad_type)]
        self.model_a = nn.Sequential(*self.model)

        # Content Encoder B
        self.model = []
        self.model += [Conv2dBlock(input_dim, dim, 7, 1, 3, norm='none', activation='none', pad_type=pad_type)]
        self.model += [nn.LeakyReLU(inplace=True)]
        current_dim = dim

        for i in range(n_downsample):
            self.model += [Conv2dBlock(current_dim, 2 * current_dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
            current_dim *= 2
        self.prelayer_b = nn.Sequential(*self.model)
        self.model = []
        for i in range(mid_downsample):
            self.model += [Conv2dBlock(current_dim, current_dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]

        self.model += [ResBlocks(n_res, current_dim, norm=norm, activation=activ, pad_type=pad_type)]
        self.model_b = nn.Sequential(*self.model)

        # Share residual block
        self.share_layer = [ResBlock(current_dim, norm=norm, activation='lrelu', pad_type=pad_type)]
        self.share_layer = nn.Sequential(*self.share_layer)
        self.output_dim = current_dim

    def forward(self, x, model_type):
        if model_type == 'a':
            prevlayer = self.prelayer_a(x)
            domain_out = self.model_a(prevlayer)
        else:
            prevlayer = self.prelayer_b(x)
            domain_out = self.model_b(prevlayer)

        share_out = self.share_layer(domain_out)
        return share_out, domain_out, prevlayer


class Decoder(nn.Module):
    def __init__(self, n_upsample, mid_downsample, n_res, dim, output_dim, res_norm='adain', activ='relu', pad_type='zero'):
        super(Decoder, self).__init__()

        self.model = []
        # AdaIN residual blocks
        self.model += [ResBlocks(n_res, dim, res_norm, activ, pad_type=pad_type)]
        # upsampling blocks
        for i in range(mid_downsample):
            self.model += [nn.Upsample(scale_factor=2),
                           Conv2dBlock(dim, dim, 5, 1, 2, norm='ln', activation=activ, pad_type=pad_type)]
        for i in range(n_upsample):
            self.model += [nn.Upsample(scale_factor=2),
                           Conv2dBlock(dim, dim // 2, 5, 1, 2, norm='ln', activation=activ, pad_type=pad_type)]
            dim //= 2
        # use reflection padding in the last conv layer
        self.model += [Conv2dBlock(dim, output_dim, 7, 1, 3, norm='none', activation='tanh', pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)


##################################################################################
# Domain Mapping
##################################################################################
class ResBlock_reverse(nn.Module):
    def __init__(self, dim, norm='in', activation='relu', pad_type='zero'):
        super(ResBlock_reverse, self).__init__()

        self.layer1 = nn.Sequential(*[Conv2dTransposeBlock(dim, dim, 3, 2, 1, 1, norm='ln', activation=activation)])
        self.layer2 = nn.Sequential(*[Conv2dBlock(dim, dim, 3, 2, 1, norm='ln', activation='none', pad_type=pad_type)])

        # twolayercnn
        # model = []
        # model += [Conv2dTransposeBlock(dim, dim, 3, 2, 1, 1, norm='ln', activation=activation)]
        # model += [Conv2dBlock(dim, dim, 3, 2, 1, norm='ln', activation='none', pad_type=pad_type)]
        # self.model = nn.Sequential(*model)

    def forward(self, x, pre_x):
        out = self.layer1(x)
        out = out + pre_x
        return self.layer2(out)
        # return self.model(x)


##################################################################################
# Sequential Models
##################################################################################
class ResBlocks(nn.Module):
    def __init__(self, num_blocks, dim, norm='in', activation='relu', pad_type='zero'):
        super(ResBlocks, self).__init__()
        self.model = []
        for i in range(num_blocks):
            self.model += [ResBlock(dim, norm=norm, activation=activation, pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, dim, n_blk, norm='none', activ='relu'):

        super(MLP, self).__init__()
        self.model = []
        self.model += [LinearBlock(input_dim, dim, norm=norm, activation=activ)]
        for i in range(n_blk - 2):
            self.model += [LinearBlock(dim, dim, norm=norm, activation=activ)]
        self.model += [LinearBlock(dim, output_dim, norm='none', activation='none')] # no output activations
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x.view(x.size(0), -1))


##################################################################################
# Basic Blocks
##################################################################################
class ResBlock(nn.Module):
    def __init__(self, dim, norm='in', activation='relu', pad_type='zero'):
        super(ResBlock, self).__init__()

        model = []
        model += [Conv2dBlock(dim ,dim, 3, 1, 1, norm=norm, activation=activation, pad_type=pad_type)]
        model += [Conv2dBlock(dim ,dim, 3, 1, 1, norm=norm, activation='none', pad_type=pad_type)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        residual = x
        out = self.model(x)
        out += residual
        return out


class BasicBlock(nn.Module):
    def __init__(self, input_dim, output_dim, norm='none', activation='lrelu', pad_type='reflect'):
        super(BasicBlock, self).__init__()

        model = []
        model += [nn.LeakyReLU(0.2)]
        model += [Conv2dBlock(input_dim ,input_dim, 3, 1, 1, norm=norm, activation='none', pad_type=pad_type)]
        model += [nn.LeakyReLU(0.2)]
        model += [Conv2dBlock(input_dim ,output_dim, 3, 1, 1, norm=norm, activation='none', pad_type=pad_type)]
        model += [nn.AvgPool2d(kernel_size=2, stride=2)]
        self.model = nn.Sequential(*model)

        model = []
        model += [nn.AvgPool2d(kernel_size=2, stride=2)]
        model += [Conv2dBlock(input_dim ,output_dim, 1, 1, 0, norm=norm, activation='none', pad_type=pad_type)]
        self.shortcut = nn.Sequential(*model)

    def forward(self, x):
        out = self.model(x)
        shortcut = self.shortcut(x)
        return out + shortcut
