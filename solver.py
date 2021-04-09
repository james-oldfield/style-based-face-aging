from model import Encoder, Decoder
from model import Classifier
import torch.nn as nn
from torchvision.utils import save_image, make_grid
import torch
import torch.nn.functional as F
import numpy as np
import os
import time
import datetime


class Solver(object):
    """Solver for training and testing StarGAN."""

    def __init__(self, loader, transfer_loader, num_transfer_images, config):
        """Initialize configurations."""

        # Data loader.
        self.loader = loader
        # self.val_loader = val_loader
        self.transfer_loader = transfer_loader
        self.num_transfer_images = num_transfer_images

        # Model configurations.
        self.image_size = config.image_size
        self.g_conv_dim = config.g_conv_dim
        self.d_conv_dim = config.d_conv_dim
        self.g_repeat_num = config.g_repeat_num
        self.lambda_rec = config.lambda_rec
        self.lambda_adv = config.lambda_adv
        self.lambda_gp = config.lambda_gp
        self.lambda_id = config.lambda_id

        self.num_classes = config.num_classes

        # Training configurations.
        self.batch_size = config.batch_size
        self.num_iters = config.num_iters
        self.num_iters_decay = config.num_iters_decay
        self.g_lr = config.g_lr
        self.c_lr = config.c_lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.resume_iters = config.resume_iters
        self.use_multiple_gpus = config.use_multiple_gpus
        self.num_layers_to_skip = config.num_layers_to_skip

        # Miscellaneous.
        self.use_tensorboard = config.use_tensorboard
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('GPU?: {}'.format(torch.cuda.is_available()))

        # Directories.
        self.log_dir = os.path.join(config.output_dir, 'logs')
        self.sample_dir = os.path.join(config.output_dir, 'samples')
        self.model_save_dir = os.path.join(config.output_dir, 'models')
        self.results_dir = os.path.join(config.output_dir, 'results')
        self.output_dir = config.output_dir
        self.image_dir = config.image_dir

        # Step size.
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step
        self.lr_update_step = config.lr_update_step

        # Build the model and tensorboard.
        self.build_model()

        if self.use_tensorboard:
            self.build_tensorboard()

    def build_model(self):
        """Create a generator and a discriminator."""

        self.data_loader = self.loader

        data_iter = iter(self.data_loader)
        trn_iter = iter(self.transfer_loader)

        self.num_sample = 16

        self.x_targets = next(trn_iter)
        self.x_targets = self.x_targets.to(self.device)

        self.x_fixed, _, _ = next(data_iter)
        self.x_fixed = self.x_fixed[:self.num_sample].to(self.device)

        self.Encoder = Encoder(self.g_conv_dim, self.g_repeat_num)
        self.Decoder = Decoder(self.num_layers_to_skip)

        self.C = Classifier(self.num_classes, self.d_conv_dim)

        if torch.cuda.device_count() > 1 and self.use_multiple_gpus:
            print('----------------------------')
            print('Using {} GPUs'.format(torch.cuda.device_count()))
            print('----------------------------')

            self.Encoder = nn.DataParallel(self.Encoder)
            self.Decoder = nn.DataParallel(self.Decoder)
            self.C = nn.DataParallel(self.C)

        if self.device.type == 'cuda':
            print('-------------------')
            print(torch.cuda.get_device_name(0))
            print('-------------------')

        self.C.to(self.device)
        self.Encoder.to(self.device)
        self.Decoder.to(self.device)

        self.g_optimizer = torch.optim.Adam(list(self.Encoder.parameters()) + list(self.Decoder.parameters()), self.g_lr, [self.beta1, self.beta2])
        self.c_optimizer = torch.optim.Adam(self.C.parameters(), self.c_lr, [self.beta1, self.beta2])

        self.print_network(self.Encoder, 'Encoder')
        self.print_network(self.Decoder, 'Decoder')
        self.print_network(self.C, 'Classifier')

        print('--------------------------------------------------------')

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    def restore_model(self, resume_iters):
        """Restore the trained generator, discriminator, and classifiers."""
        print('Loading the trained models from step {}...'.format(resume_iters))
        Enc_path = os.path.join(self.model_save_dir, '{}-Enc.ckpt'.format(resume_iters))
        Dec_path = os.path.join(self.model_save_dir, '{}-Dec.ckpt'.format(resume_iters))
        C_path = os.path.join(self.model_save_dir, '{}-C.ckpt'.format(resume_iters))

        self.Encoder.load_state_dict(torch.load(Enc_path, map_location=lambda storage, loc: storage))
        self.Decoder.load_state_dict(torch.load(Dec_path, map_location=lambda storage, loc: storage))
        self.C.load_state_dict(torch.load(C_path, map_location=lambda storage, loc: storage))

    def build_tensorboard(self):
        """Build a tensorboard logger."""
        from logger import Logger
        self.logger = Logger(self.log_dir)

    def update_lr(self, g_lr, c_lr):
        """Decay learning rates of the generator and discriminator."""
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = g_lr

        for param_group in self.c_optimizer.param_groups:
            param_group['lr'] = c_lr

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()
        self.c_optimizer.zero_grad()

    def denorm(self, x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def gradient_penalty(self, y, x):
        dydx = torch.autograd.grad(outputs=y.mean(),
                                   inputs=x,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        grad_dout2 = dydx.pow(2)
        assert (grad_dout2.size() == x.size())
        reg = grad_dout2.sum() / x.size(0)
        return reg

    def train(self):
        # Learning rate cache for decaying.
        g_lr = self.g_lr
        c_lr = self.c_lr

        # Start training from scratch or resume training.
        start_iters = 0
        if self.resume_iters:
            start_iters = self.resume_iters
            self.restore_model(self.resume_iters)

        # Start training.
        print('Start training...')
        start_time = time.time()

        for i in range(start_iters, self.num_iters):
            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #

            # Fetch real images and labels.
            try:
                x_real, label_org, label_org_idx = next(self.data_iter)
            except:
                self.data_iter = iter(self.data_loader)
                x_real, label_org, label_org_idx = next(self.data_iter)

            # generate targets randomly
            trg_idx = torch.randperm(self.batch_size).to(self.device)
            cyc_idx = torch.argsort(trg_idx).to(self.device)

            x_real = x_real.to(self.device)
            label_org = label_org.to(self.device)
            label_org_idx = label_org_idx.to(self.device)

            label_trg_idx = label_org_idx[trg_idx]

            # =================================================================================== #
            #                             2. Train the discriminator                              #
            # =================================================================================== #

            real_cls_logits_real = self.C(x_real, label_org_idx)[1]
            c_loss_real = F.binary_cross_entropy_with_logits(real_cls_logits_real, torch.ones_like(real_cls_logits_real))

            # get features for each image in the batch
            cls_features = [[layer.mean([2, 3]), layer.std([2, 3])]
                            for layer in self.C(torch.stack([x_real[idx] for idx in trg_idx]))[0][:-1]]

            z_fake = self.Encoder(x_real)
            x_fake = self.Decoder(z_fake, cls_features)

            fake_cls_logits_real = self.C(x_fake.detach(), label_trg_idx)[1]
            c_loss_fake = F.binary_cross_entropy_with_logits(fake_cls_logits_real, torch.zeros_like(fake_cls_logits_real))

            # align real images with the label of the transfers of x_fake
            x_shuf = x_real[trg_idx]
            x_shuf.requires_grad_()

            out_src = self.C(x_shuf, label_trg_idx)[1]
            c_loss_gp = self.gradient_penalty(out_src, x_shuf)

            # Logging.
            loss = {}

            # Backward and optimize.
            c_loss = c_loss_real + c_loss_fake + self.lambda_gp * c_loss_gp
            self.reset_grad()
            c_loss.backward()
            self.c_optimizer.step()

            loss['C/loss_real'] = c_loss_real.item()
            loss['C/loss_fake'] = c_loss_fake.item()
            loss['C/loss_gp'] = c_loss_gp.item()

            # =================================================================================== #
            #                               3. Train the generator                                #
            # =================================================================================== #

            cls_features = [[layer.mean([2, 3]), layer.std([2, 3])]
                            for layer in self.C(torch.stack([x_real[idx] for idx in trg_idx]))[0][:-1]]

            z_fake = self.Encoder(x_real)
            x_trn = self.Decoder(z_fake, cls_features)

            # get the classifier features for original labels (cycle constraint)
            rec_features = [[layer.mean([2, 3]), layer.std([2, 3])]
                            for layer in self.C(x_trn[cyc_idx])[0][:-1]]

            # cycle constraint
            x_rec = self.Decoder(self.Encoder(x_trn), rec_features)

            fake_cls_logits_fake = self.C(x_trn, label_trg_idx)[1]
            real_cls_logits_real = self.C(x_real, label_org_idx)[1].detach()

            g_loss_fake = torch.mean((real_cls_logits_real[trg_idx] - fake_cls_logits_fake) ** 2)
            g_loss_rec = torch.mean(torch.abs(x_real - x_rec))
            g_loss_id = torch.mean(torch.abs(x_real - x_trn))

            # Backward and optimize.
            g_loss = self.lambda_adv * g_loss_fake + self.lambda_rec * g_loss_rec + self.lambda_id * g_loss_id

            self.reset_grad()
            g_loss.backward()
            self.g_optimizer.step()

            # Logging.
            loss['G/loss_rec'] = g_loss_rec.item()
            loss['G/loss_fake'] = g_loss_fake.item()

            # breakpoint()

            # =================================================================================== #
            #                                 5. Miscellaneous                                    #
            # =================================================================================== #

            # Print out training information.
            if (i + 1) % self.log_step == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}]".format(et, i + 1, self.num_iters)
                for tag, value in loss.items():
                    log += ", {}: {:.4f}".format(tag, value)
                print(log)

                if self.use_tensorboard:
                    for tag, value in loss.items():
                        self.logger.scalar_summary(tag, value, i + 1)

            # Translate fixed images for debugging.
            if (i + 1) % self.sample_step == 0:
                with torch.no_grad():

                    x_fake_list = [self.x_fixed]

                    for t_i, target in enumerate(self.x_targets):
                        target_features = [[layer.mean([2, 3]).repeat(self.num_sample, 1), layer.std([2, 3]).repeat(self.num_sample, 1)]
                                           for layer in self.C(target.unsqueeze(0))[0][:-1]]

                        z_fake = self.Encoder(self.x_fixed)
                        x_fake_list.append(self.Decoder(z_fake, target_features))

                    x_concat = torch.cat(x_fake_list, dim=3)
                    inp = make_grid(torch.cat([torch.zeros((1, 3, 128, 128)).to(self.device) - 1, self.x_targets], 0), padding=0).unsqueeze(0)
                    x_concat = torch.cat([inp, x_concat], 0)

                    sample_path = os.path.join(self.sample_dir, '{}-images.jpg'.format(i + 1))
                    save_image(self.denorm(x_concat.data.cpu()), sample_path, nrow=1, padding=0)
                    print('Saved real and fake images to {}...'.format(sample_path))

            # Save model checkpoints.
            if (i + 1) % self.model_save_step == 0:
                Enc_path = os.path.join(self.model_save_dir, '{}-Enc.ckpt'.format(i + 1))
                Dec_path = os.path.join(self.model_save_dir, '{}-Dec.ckpt'.format(i + 1))
                C_path = os.path.join(self.model_save_dir, '{}-C.ckpt'.format(i + 1))
                torch.save(self.Encoder.state_dict(), Enc_path)
                torch.save(self.Decoder.state_dict(), Dec_path)
                torch.save(self.C.state_dict(), C_path)
                print('Saved model checkpoints into {}...'.format(self.model_save_dir))

            # Decay learning rates.
            if (i + 1) % self.lr_update_step == 0 and (i + 1) > (self.num_iters - self.num_iters_decay):
                g_lr -= (self.g_lr / float(self.num_iters_decay))
                c_lr -= (self.c_lr / float(self.num_iters_decay))
                self.update_lr(g_lr, c_lr)
                print('Decayed learning rates, g_lr: {}, c_lr: {}.'.format(g_lr, c_lr))
