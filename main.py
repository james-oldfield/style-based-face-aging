import os
from shutil import copyfile
import argparse
from solver import Solver
from data_loader import get_loader
from torch.backends import cudnn
import json


def str2bool(v):
    return v.lower() in ('true')


def main(config):
    # For fast training.
    cudnn.benchmark = True

    # Create directories if not exist.
    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)
    if not os.path.exists(os.path.join(config.output_dir, 'logs')):
        os.makedirs(os.path.join(config.output_dir, 'logs'))

    if not os.path.exists(os.path.join(config.output_dir, 'models')):
        os.makedirs(os.path.join(config.output_dir, 'models'))
    if not os.path.exists(os.path.join(config.output_dir, 'samples')):
        os.makedirs(os.path.join(config.output_dir, 'samples'))
    if not os.path.exists(os.path.join(config.output_dir, 'results')):
        os.makedirs(os.path.join(config.output_dir, 'results'))
    if not os.path.exists(os.path.join(config.output_dir, 'source-code')):
        os.makedirs(os.path.join(config.output_dir, 'source-code'))

    with open(os.path.join(config.output_dir, 'source-code/hyperparams.json'), 'w') as fp:
        json.dump(vars(config), fp)

    # make dirs to store FID samples
    if not os.path.exists(os.path.join(config.output_dir, 'fid-samples')):
        os.makedirs(os.path.join(config.output_dir, 'fid-samples'))

    copyfile('main.py', os.path.join(config.output_dir, 'source-code', 'main.py'))
    copyfile('solver.py', os.path.join(config.output_dir, 'source-code', 'solver.py'))
    copyfile('model.py', os.path.join(config.output_dir, 'source-code', 'model.py'))
    copyfile('data_loader.py', os.path.join(config.output_dir, 'source-code', 'data_loader.py'))

    num_transfer_images = 4

    # Data loaders
    train_loader = get_loader(config.image_dir, config.image_size, config.batch_size, config.num_classes, config.mode, config.num_workers)
    transfer_loader = get_loader(config.image_dir, config.image_size, num_transfer_images, config.num_classes, 'transfer', config.num_workers) 

    solver = Solver(train_loader, transfer_loader, num_transfer_images, config)

    if config.mode == 'train':
        solver.train()
    elif config.mode == 'test':
        solver.test()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model configuration.
    parser.add_argument('--image_size', type=int, default=128, help='image resolution')
    parser.add_argument('--g_conv_dim', type=int, default=64, help='number of conv filters in the first layer of G')
    parser.add_argument('--d_conv_dim', type=int, default=64, help='number of conv filters in the first layer of D')
    parser.add_argument('--g_repeat_num', type=int, default=6, help='number of residual blocks in G')
    parser.add_argument('--lambda_rec', type=float, default=0.01, help='weight for reconstruction loss')
    parser.add_argument('--lambda_adv', type=float, default=1.0, help='weight for reconstruction loss')
    parser.add_argument('--lambda_gp', type=float, default=10, help='weight for gradient penalty')
    parser.add_argument('--lambda_id', type=float, default=10e-04, help='weight for identity preservation loss term')

    parser.add_argument('--num_classes', type=int, default=4, help='num age classes')

    # Training configuration.
    parser.add_argument('--batch_size', type=int, default=16, help='mini-batch size')
    parser.add_argument('--num_iters', type=int, default=100000, help='number of total iterations for training D+G')
    parser.add_argument('--num_iters_decay', type=int, default=100000, help='number of iterations for decaying lr')
    parser.add_argument('--g_lr', type=float, default=0.0001, help='learning rate for G')
    parser.add_argument('--c_lr', type=float, default=0.0001, help='learning rate for C')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
    parser.add_argument('--resume_iters', type=int, default=None, help='resume training from this step')
    parser.add_argument('--use_multiple_gpus', type=str2bool, default=False, help='use multiple gpus')
    parser.add_argument('--num_layers_to_skip', type=int, default=1, help='Num of skip connections to omit')

    # Miscellaneous.
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--use_tensorboard', type=str2bool, default=False)

    # Directories.
    parser.add_argument('--image_dir', type=str, default='./data-cacd/')
    parser.add_argument('--output_dir', type=str, default='fair-gan')

    # Step size.
    parser.add_argument('--log_step', type=int, default=1)
    parser.add_argument('--sample_step', type=int, default=500)
    parser.add_argument('--model_save_step', type=int, default=10000)
    parser.add_argument('--lr_update_step', type=int, default=1000)

    config = parser.parse_args()
    print(config)
    main(config)
