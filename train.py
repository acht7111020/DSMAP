"""
Copyright (C) 2020 Hsin-Yu Chang <acht7111020@gmail.com>
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import argparse
import sys
import shutil
import os
import pathlib

import torch.backends.cudnn as cudnn
import torch
import tensorboardX

from utils import get_all_data_loaders, get_config, write_loss, write_2images, Timer
from trainer import DSMAP_Trainer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/cat2dog.yaml', help='Path to the config file')
    parser.add_argument('--save_name', type=str, default='.', help="Name to save the training results (ex. DS-ver1)")
    parser.add_argument("--resume", action="store_true")
    opts = parser.parse_args()

    cudnn.benchmark = True

    # Load experiment setting
    config = get_config(opts.config)
    max_iter = config['max_iter']
    display_size = config['display_size']

    # Setup model and data loader
    trainer = DSMAP_Trainer(config)
    trainer.build_optimizer(config)

    trainer.cuda()
    train_loader_a, train_loader_b, test_loader_a, test_loader_b = get_all_data_loaders(config)
    train_display_images_a = torch.stack([train_loader_a.dataset[i] for i in range(display_size)]).cuda()
    train_display_images_b = torch.stack([train_loader_b.dataset[i] for i in range(display_size)]).cuda()
    test_display_images_a = torch.stack([test_loader_a.dataset[i] for i in range(display_size)]).cuda()
    test_display_images_b = torch.stack([test_loader_b.dataset[i] for i in range(display_size)]).cuda()

    # Setup logger and output folders
    data_name = os.path.splitext(os.path.basename(opts.config))[0]
    log_directory = os.path.join("./logs", data_name, opts.save_name)
    output_directory = os.path.join("./outputs", data_name, opts.save_name)
    image_directory = os.path.join(output_directory, 'images')
    checkpoint_directory = os.path.join(output_directory, 'ckpts')
    pathlib.Path(log_directory).mkdir(parents=True, exist_ok=True)
    pathlib.Path(image_directory).mkdir(parents=True, exist_ok=True)
    pathlib.Path(checkpoint_directory).mkdir(parents=True, exist_ok=True)

    train_writer = tensorboardX.SummaryWriter(log_directory)
    shutil.copy(opts.config, os.path.join(output_directory, 'config.yaml')) # copy config file to output folder

    # Start training
    iterations = trainer.resume(checkpoint_directory, hyperparameters=config) if opts.resume else 0
    while True:
        for it, (images_a, images_b) in enumerate(zip(train_loader_a, train_loader_b)):
            trainer.update_learning_rate()
            images_a, images_b = images_a.cuda().detach(), images_b.cuda().detach()

            with Timer("Elapsed time in update: %f"):
                # Main training code
                trainer.dis_update(images_a, images_b, config, iterations)
                trainer.gen_update(images_a, images_b, config, iterations)
                torch.cuda.synchronize()

            # Dump training stats in log file
            if (iterations + 1) % config['log_iter'] == 0:
                print("Iteration: %08d/%08d" % (iterations + 1, max_iter))
                write_loss(iterations, trainer, train_writer)

            # Write images
            if (iterations + 1) == 1 or (iterations + 1) % config['image_save_iter'] == 0:
                with torch.no_grad():
                    test_image_outputs = trainer.sample(test_display_images_a, test_display_images_b)
                    train_image_outputs = trainer.sample(train_display_images_a, train_display_images_b)
                write_2images(test_image_outputs, display_size, image_directory, 'test_%08d' % (iterations + 1))
                write_2images(train_image_outputs, display_size, image_directory, 'train_%08d' % (iterations + 1))

            if (iterations + 1) % config['image_display_iter'] == 0 or iterations == 0:
                with torch.no_grad():
                    image_outputs = trainer.sample(train_display_images_a, train_display_images_b)
                write_2images(image_outputs, display_size, image_directory, 'train_current')

            # Save network weights
            if (iterations + 1) % config['snapshot_save_iter'] == 0:
                trainer.save(checkpoint_directory, iterations)

            iterations += 1
            if iterations >= max_iter:
                sys.exit('Finish training')
