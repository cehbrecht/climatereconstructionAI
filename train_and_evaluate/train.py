import os
import torch
import sys

sys.path.append('./')

from model.loss import GeneratorLoss, DiscriminatorLoss
from tensorboardX import SummaryWriter
from torch.utils import data
from tqdm import tqdm
from model.Generator import PConvLSTMGenerator
from model.Discriminator import Discriminator
from utils.io import load_ckpt, save_ckpt
from utils.netcdfloader import NetCDFLoader, InfiniteSampler
from utils.evaluation import create_snapshot_image
import config as cfg

cfg.set_train_args()

if not os.path.exists(cfg.snapshot_dir):
    os.makedirs('{:s}/images'.format(cfg.snapshot_dir))
    os.makedirs('{:s}/ckpt'.format(cfg.snapshot_dir))

if not os.path.exists(cfg.log_dir):
    os.makedirs(cfg.log_dir)
writer = SummaryWriter(log_dir=cfg.log_dir)

# define data set + iterator
dataset_train = NetCDFLoader(cfg.data_root_dir, cfg.img_names, cfg.mask_dir, cfg.mask_names, 'train', cfg.data_types,
                             cfg.lstm_steps, cfg.prev_next_steps)
dataset_val = NetCDFLoader(cfg.data_root_dir, cfg.img_names, cfg.mask_dir, cfg.mask_names, 'val', cfg.data_types,
                           cfg.lstm_steps, cfg.prev_next_steps)
iterator_train = iter(data.DataLoader(dataset_train, batch_size=cfg.batch_size,
                                      sampler=InfiniteSampler(len(dataset_train)),
                                      num_workers=cfg.n_threads))

# define network model
lstm = True
if cfg.lstm_steps == 0:
    lstm = False

generator = PConvLSTMGenerator(image_size=cfg.image_size,
                               num_enc_dec_layers=cfg.encoding_layers,
                               num_pool_layers=cfg.pooling_layers,
                               num_in_channels=len(cfg.data_types) * (2 * cfg.prev_next_steps + 1),
                               num_out_channels=cfg.out_channels,
                               lstm=lstm).to(cfg.device)

discriminator = Discriminator(image_size=cfg.image_size,
                              num_enc_dec_layers=cfg.encoding_layers,
                              num_pool_layers=cfg.pooling_layers,
                              num_in_channels=2).to(cfg.device)

# define learning rate
if cfg.finetune:
    lr = cfg.lr_finetune
    generator.freeze_enc_bn = True
else:
    lr = cfg.lr

# define optimizer and loss functions
generator_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, generator.parameters()), lr=lr, betas=(0.5, 0.99))
discriminator_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, discriminator.parameters()), lr=lr, betas=(0.5, 0.99))

generator_criterion = GeneratorLoss().to(cfg.device)
discriminator_criterion = DiscriminatorLoss().to(cfg.device)

# define start point
start_iter = 0
if cfg.resume_iter:
    # load generator
    start_iter = load_ckpt('{}/ckpt/generator_{}'.format(cfg.snapshot_dir, cfg.resume_iter),
                           [('model', generator)], cfg.device, [('optimizer', generator_optimizer)])
    # load discriminator
    load_ckpt('{}/ckpt/discriminator_{}'.format(cfg.snapshot_dir, cfg.resume_iter),
              [('model', discriminator)], cfg.device, [('optimizer', discriminator_optimizer)])
    for param_group in generator_optimizer.param_groups:
        param_group['lr'] = lr
    for param_group in discriminator_optimizer.param_groups:
        param_group['lr'] = lr
    print('Starting from iter ', start_iter)

for i in tqdm(range(start_iter, cfg.max_iter)):
    # train model
    generator.train()
    discriminator.train()

    image, mask, gt = [x.to(cfg.device) for x in next(iterator_train)]

    discr_gt = discriminator(gt[:, 0, :, :, :], mask[:, 0, :, :, :])
    output = generator(image, mask)[:, 0, :, :, :]
    discr_output = discriminator(output.detach(), mask[:, 0, :, :, :])

    # discriminator loss
    discriminator.zero_grad()
    discriminator_loss = discriminator_criterion(discr_gt, discr_output)
    discriminator_loss.backward()
    discriminator_optimizer.step()

    # generator loss
    generator.zero_grad()
    generator_loss = 0.0
    loss_dict = generator_criterion(mask[:, 0, :, :, :], output, gt[:, 0, :, :, :], discr_output.detach())
    for key, coef in cfg.LAMBDA_DICT_IMG_INPAINTING.items():
        value = coef * loss_dict[key]
        generator_loss += value
    generator_loss.backward()
    generator_optimizer.step()

    # save checkpoint
    if (i + 1) % cfg.save_model_interval == 0 or (i + 1) == cfg.max_iter:
        save_ckpt('{:s}/ckpt/generator_{:d}.pth'.format(cfg.snapshot_dir, i + 1),
                  [('model', generator)], [('optimizer', generator_optimizer)], i + 1)
        save_ckpt('{:s}/ckpt/discriminator_{:d}.pth'.format(cfg.snapshot_dir, i + 1),
                  [('model', discriminator)], [('optimizer', discriminator_optimizer)], i + 1)

    # create snapshot image
    if cfg.log_interval and (i + 1) % cfg.log_interval == 0:
        generator.eval()
        create_snapshot_image(generator, dataset_val, '{:s}/images/test_{:d}'.format(cfg.snapshot_dir, i + 1),
                              cfg.lstm_steps)

writer.close()
