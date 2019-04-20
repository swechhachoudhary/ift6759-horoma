import sys
import os
import numpy as np
import math
import torch
from torchvision.transforms import functional
import models.HALI as hali
import models.ALI as ali
from torch.optim.optimizer import Optimizer, required
from utils.model_utils import get_ae_dataloaders
from torch.autograd import Variable
from torch import nn
from torch import Tensor
from torch.nn import Parameter
from utils.custom_optimizers import *
from itertools import chain
from torchvision.utils import save_image
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import re
from models.clustering import *
from utils.utils import *
import copy
from random import randint
import numpy as np
from os import listdir
from torch.nn import init
from os.path import isfile, isdir, join


def initialize_ali(configs, data):
    """
    initialize_ali is a function for initializing all models and loaders needed in training
    :param configs: a dictionary with all params necessary for training
    :param data: Data used to create the loader.
    :return Gx: Decoder/Generator
    :return Gz: Encoder
    :return Disc: Discriminator
    :return optim_g: Optimizer for Generator params
    :return optim_d: Optimizer for Discriminator params
    :return train_loader: the dataloader we will use for training
    :return cuda: whether we are running on cuda
    :return configs: updated with correct model and image path
    """

    IMAGE_PATH = '../experiments/' + configs['experiment'] + '/images'
    MODEL_PATH = '../experiments/' + configs['experiment'] + '/models'
    configs['IMAGE_PATH'] = IMAGE_PATH
    configs['MODEL_PATH'] = MODEL_PATH

    if not os.path.exists(IMAGE_PATH):
        print('mkdir ', IMAGE_PATH)
        os.mkdir(IMAGE_PATH)
    if not os.path.exists(MODEL_PATH):
        print('mkdir ', MODEL_PATH)
        os.mkdir(MODEL_PATH)

    Zdim = configs['Zdim']
    BS = configs['batch_size']

    Gx = ali.GeneratorX(zd=Zdim, ch=3)

    Gz = ali.GeneratorZ(zd=Zdim, ch=3)

    Disc = ali.Discriminator(ch=3, zd=Zdim)

    if 'continue_from' in configs:

        if configs['continue_from'] == -1:

            start_epoch = get_max_epoch(configs) - 1

            Gx.load_state_dict(torch.load(
                MODEL_PATH + '/Gx-' + str(start_epoch) + '.pth'))
            Gz.load_state_dict(torch.load(
                MODEL_PATH + '/Gz-' + str(start_epoch) + '.pth'))
            Disc.load_state_dict(torch.load(
                MODEL_PATH + '/Dict-' + str(start_epoch) + '.pth'))

        else:

            Gx.load_state_dict(torch.load(
                MODEL_PATH + '/Gx-' + str(configs['continue_from']) + '.pth'))
            Gz.load_state_dict(torch.load(
                MODEL_PATH + '/Gz-' + str(configs['continue_from']) + '.pth'))
            Disc.load_state_dict(torch.load(
                MODEL_PATH + '/Dict-' + str(configs['continue_from']) + '.pth'))

    cuda = True if torch.cuda.is_available() else False

    if cuda:
        Gx.cuda()
        Gz.cuda()
        Disc.cuda()

    gen = chain(Gx.parameters(), Gz.parameters())

    decay = 0
    beta1 = 0.5
    beta2 = 0.999
    amsgrad = False

    if 'decay' in configs:
        decay = configs['decay']
    if 'amsgrad' in configs:
        amsgrad = configs['amsgrad']
    if ('beta1' in configs) and ('beta2' in configs):
        beta1 = configs['beta1']
        beta2 = configs['beta2']

    if configs['optim'] == 'Adam':

        optim_d = torch.optim.Adam(Disc.parameters(), lr=configs[
                                   'lr_d'], betas=(beta1, beta2), weight_decay=decay)
        optim_g = torch.optim.Adam(gen, configs['lr_g'], betas=(
            beta1, beta2), weight_decay=decay)

    elif configs['optim'] == 'OAdam':

        optim_d = OAdam(Disc.parameters(), lr=configs['lr_d'], betas=(
            beta1, beta2), weight_decay=decay, amsgrad=amsgrad)
        optim_g = OAdam(gen, configs['lr_g'], betas=(
            beta1, beta2), weight_decay=decay, amsgrad=amsgrad)

    elif configs['optim'] == 'OMD':

        optim_d = OptMirrorAdam(Disc.parameters(), lr=configs['lr_d'], betas=(
            beta1, beta2), weight_decay=decay, amsgrad=amsgrad)
        optim_g = OptMirrorAdam(gen, configs['lr_g'], betas=(
            beta1, beta2), weight_decay=decay, amsgrad=amsgrad)

    train_loader = DataLoader(data, batch_size=BS, shuffle=True)

    return Gx, Gz, Disc, optim_g, optim_d, train_loader, cuda, configs


def train_epoch_ali(Gz, Gx, Disc, optim_d, optim_g, loader, epoch, cuda, configs):
    """
    :param Gz: Encoder
    :param Gx: Decoder/Generator
    :param Disc: Discriminator
    :param optim_d: Optimizer for Discriminator params
    :param optim_g: Optimizer for Generator params
    :param loader: data loader
    :param epoch: number of current epoch
    :param cuda: whether we are running on cuda
    :param configs: a dictionary with all params necessary for training
    :return g_loss: Generator loss
    :return d_loss: Decoder loss
    :return d_true: Prediction for x
    :return d_false: Prediction for x_hat
    """

    ncritic = configs['n_critic']
    cnt = 0
    gcnt = 0
    df = 0
    dt = 0
    dl = 0
    gl = 0

    for i, (imgs) in enumerate(loader):

        loss_d = runloop_d_ali(imgs, Gx, Gz, Disc, optim_d, cuda, configs)

        if i % ncritic == 0:
            loss_g, d_true, d_fake = runloop_g_ali(
                imgs, Gx, Gz, Disc, optim_g, cuda, configs)

            gl = gl + loss_g
            df = df + d_fake.item()
            dt = dt + d_true.data.mean().item()
            gcnt = gcnt + 1

        cnt = cnt + 1
        dl = dl + loss_d

    g_loss = gl / gcnt
    d_loss = dl / cnt

    d_true = dt / gcnt
    d_false = df / gcnt

    return g_loss, d_loss, d_true, d_false


def training_loop_ali(Gz, Gx, Disc, optim_d, optim_g, train_loader, configs, experiment, cuda):
    """
    :param Gz: Encoder
    :param Gx: Decoder/Generator
    :param Disc: Discriminator
    :param optim_d: Optimizer for Discriminator params
    :param optim_g: Optimizer for Generator params
    :param loader: data loader
    :param configs: a dictionary with all params necessary for training
    :param experiment: comet_ml experiment variable to store results
    :param cuda: whether we are running on cuda
    """

    Zdim = configs['Zdim']
    if 'continue_from' in configs:
        if configs['continue_from'] == -1:
            start_epoch = get_max_epoch(configs) - 1
            end_epoch = start_epoch + configs['n_epochs']
        else:
            start_epoch = configs['continue_from']
            end_epoch = start_epoch + configs['n_epochs']
    else:
        start_epoch = 0
        end_epoch = configs['n_epochs']

    if 'lr_scheduler' in configs:
        if configs['lr_scheduler'] == 'cyclic':
            milestones = [int(np.floor((2**x) * 1.2)) for x in range(7)]
            if configs['scheduler_mode'] == 'both':
                scheduler1 = CyclicCosAnnealingLR(
                    optim_d, milestones=milestones, eta_min=1e-5)
                scheduler2 = CyclicCosAnnealingLR(
                    optim_g, milestones=milestones, eta_min=1e-5)
            else:
                scheduler1 = CyclicCosAnnealingLR(
                    optim_d, milestones=milestones, eta_min=1e-5)

        if 'continue_from' in configs and configs['continue_from'] == -1:
            optim_d.load_state_dict(torch.load(
                configs['MODEL_PATH'], 'optim_d.pth'))
            optim_g.load_state_dict(torch.load(
                configs['MODEL_PATH'], 'optim_g.pth'))
            scheduler1.load_state_dict(torch.load(
                os.path.join(configs['MODEL_PATH'], 'scheduler1.pth')))
            if configs['scheduler_mode'] == 'both':
                scheduler1.load_state_dict(torch.load(
                    os.path.join(configs['MODEL_PATH'], 'scheduler2.pth')))

    for epoch in range(start_epoch, end_epoch):

        if 'lr_scheduler' in configs:
            scheduler1.step()
            if configs['scheduler_mode'] == 'both':
                scheduler2.step()

        g_loss, d_loss, d_true, d_false = train_epoch_ali(
            Gz, Gx, Disc, optim_d, optim_g, train_loader, epoch, cuda, configs
        )

        if 'lr_scheduler' in configs:
            torch.save(scheduler1.state_dict(), os.path.join(
                configs['MODEL_PATH'], 'scheduler1.pth'))
            if configs['scheduler_mode'] == 'both':
                torch.save(scheduler1.state_dict(), os.path.join(
                    configs['MODEL_PATH'], 'scheduler2.pth'))
        torch.save(optim_d.state_dict(), os.path.join(
            configs['MODEL_PATH'], 'optim_d.pth'))
        torch.save(optim_g.state_dict(), os.path.join(
            configs['MODEL_PATH'], 'optim_g.pth'))

        save_models_ali(Gz, Gx, Disc, configs['MODEL_PATH'], epoch)
        sys.stdout.write("\r[%5d / %5d]: G: %.4f D: %.4f D(x,Gz(x)): %.4f D(Gx(z),z): %.4f" %
                         (epoch, configs['n_epochs'], g_loss, d_loss, d_true, d_false))

        experiment.log_metric('g_loss', g_loss)
        experiment.log_metric('d_loss', d_loss)
        experiment.log_metric('d_true', d_true)
        experiment.log_metric('d_fake', d_false)

        print()


def runloop_g_ali(imgs, Gx, Gz, Disc, optim_g, cuda, configs):
    """
    :param imgs: data for generator loop
    :param Gz: Encoder
    :param Gx: Decoder/Generator
    :param Disc: Discriminator
    :param optim_g: Optimizer for Generator params
    :param cuda: whether we are running on cuda
    :param configs: a dictionary with all params necessary for training
    :return g_loss: Generator loss
    :return d_true: Prediction for x
    :return d_false: Prediction for x_hat
    """

    softplus = nn.Softplus()
    Zdim = configs['Zdim']

    batch_size = imgs.size(0)

    if cuda:
        imgs = imgs.cuda()
    imgs = Variable(imgs)
    batch_size = imgs.size(0)

    z = torch.FloatTensor(batch_size, Zdim, 1, 1).normal_(0, 1)

    zv = Variable(z).cuda()

    encoded1 = Gz(imgs)
    z = reparameterize(encoded1)

    imgs_fake = Gx(zv)

    def g_closure():

        Gx.zero_grad()

        Gz.zero_grad()

        d_true = Disc(imgs, z)
        d_fake = Disc(imgs_fake, zv)

        loss_g = torch.mean(softplus(d_true) + softplus(-d_fake))

        loss_g.backward(retain_graph=True)
        return loss_g.data.item(), d_fake.data.mean(), d_true.data.mean()

    loss_g, d_fake, d_true = optim_g.step(g_closure)
    return loss_g, d_true, d_fake


def runloop_d_ali(imgs, Gx, Gz, Disc, optim_d, cuda, configs):
    """
    :param imgs: data for generator loop
    :param Gz: Encoder
    :param Gx: Decoder/Generator
    :param Disc: Discriminator
    :param optim_g: Optimizer for Generator params
    :param cuda: whether we are running on cuda
    :param configs: a dictionary with all params necessary for training
    :return d_loss: Discriminator loss
    """

    softplus = nn.Softplus()
    Zdim = configs['Zdim']

    batch_size = imgs.size(0)

    if cuda:
        imgs = imgs.cuda()
    imgs = Variable(imgs)
    batch_size = imgs.size(0)

    z = torch.FloatTensor(batch_size, Zdim, 1, 1).normal_(0, 1)

    zv = Variable(z).cuda()

    encoded1 = Gz(imgs)
    z = reparameterize(encoded1)

    imgs_fake = Gx(zv)

    def d_closure():

        Disc.zero_grad()
        batch_size = imgs.size(0)

        d_true = Disc(imgs, z)
        d_fake = Disc(imgs_fake, zv)

        if configs['gp']:

            gp = calc_gradient_penalty_ali(
                Disc, imgs, imgs_fake, zv, z, configs['gp_lambda'])
            loss_d = torch.mean(softplus(-d_true) + softplus(d_fake)) + gp

        else:

            loss_d = torch.mean(softplus(-d_true) + softplus(d_fake))

        loss_d.backward(retain_graph=True)
        return loss_d.data.item()

    loss_d = optim_d.step(d_closure)

    return loss_d


def initialize_hali(configs, data):
    """
    initialize_hali is a function for initializing all models and loaders needed in training HALI

    :param configs: a dictionary with all params necessary for training
    :param data: Data used to create the loader.

    :return Gx1: Level 1 of Decoder/Generator
    :return Gx2: Level 2 of Decoder/Generator
    :return Gz1: Level 1 of Encoder
    :return Gz2: Level 2 of Encoder
    :return Disc: Discriminator
    :return optim_g: Optimizer for Generator params
    :return optim_d: Optimizer for Discriminator params
    :return train_loader: the dataloader we will use for training
    :return cuda: whether we are running on cuda
    :return configs: updated with correct model and image path

    """

    IMAGE_PATH = '../experiments/' + configs['experiment'] + '/images'
    MODEL_PATH = '../experiments/' + configs['experiment'] + '/models'

    configs['IMAGE_PATH'] = IMAGE_PATH
    configs['MODEL_PATH'] = MODEL_PATH
    if not os.path.exists(IMAGE_PATH):
        print('mkdir ', IMAGE_PATH)
        os.mkdir(IMAGE_PATH)
    if not os.path.exists(MODEL_PATH):
        print('mkdir ', MODEL_PATH)
        os.mkdir(MODEL_PATH)

    Zdim = configs['Zdim']
    zd1 = configs['z1dim']
    BS = configs['batch_size']

    Gz1 = hali.GeneratorZ1(zd=Zdim, ch=3, zd1=zd1)
    Gz2 = hali.GeneratorZ2(zd=Zdim, zd1=zd1)
    Disc = hali.Discriminator(ch=3, zd=Zdim, zd1=zd1)

    if 'genx' in configs:

        if configs['genx'] == 'interpolate':
            print('interpolate')
            Gx1 = hali.GeneratorX1_interpolate(zd=Zdim, ch=3, zd1=zd1)
            Gx2 = hali.GeneratorX2_interpolate(zd=Zdim, ch=3, zd1=zd1)
        else:
            print('convolve')
            Gx1 = hali.GeneratorX1_convolve(zd=Zdim, ch=3, zd1=zd1)
            Gx2 = hali.GeneratorX2_convolve(zd=Zdim, ch=3, zd1=zd1)
    else:

        Gx1 = hali.GeneratorX1(zd=Zdim, ch=3, zd1=zd1)
        Gx2 = hali.GeneratorX2(zd=Zdim, ch=3, zd1=zd1)

    if 'continue_from' in configs:

        if configs['continue_from'] == -1:

            start_epoch = get_max_epoch(configs) - 1

            Gx1.load_state_dict(torch.load(
                MODEL_PATH + '/Gx1-' + str(start_epoch) + '.pth'))
            Gx2.load_state_dict(torch.load(
                MODEL_PATH + '/Gx2-' + str(start_epoch) + '.pth'))
            Gz1.load_state_dict(torch.load(
                MODEL_PATH + '/Gz1-' + str(start_epoch) + '.pth'))
            Gz2.load_state_dict(torch.load(
                MODEL_PATH + '/Gz2-' + str(start_epoch) + '.pth'))
            Disc.load_state_dict(torch.load(
                MODEL_PATH + '/Disc-' + str(start_epoch) + '.pth'))

        else:

            Gx1.load_state_dict(torch.load(
                MODEL_PATH + '/Gx1-' + str(configs['continue_from']) + '.pth'))
            Gx2.load_state_dict(torch.load(
                MODEL_PATH + '/Gx2-' + str(configs['continue_from']) + '.pth'))
            Gz1.load_state_dict(torch.load(
                MODEL_PATH + '/Gz1-' + str(configs['continue_from']) + '.pth'))
            Gz2.load_state_dict(torch.load(
                MODEL_PATH + '/Gz2-' + str(configs['continue_from']) + '.pth'))
            Disc.load_state_dict(torch.load(
                MODEL_PATH + '/Disc-' + str(configs['continue_from']) + '.pth'))

    cuda = True if torch.cuda.is_available() else False

    if cuda:
        Gx1.cuda()
        Gz1.cuda()
        Gx2.cuda()
        Gz2.cuda()
        Disc.cuda()

    gen = chain(Gx1.parameters(), Gx2.parameters(),
                Gz1.parameters(), Gz2.parameters())

    decay = 0
    beta1 = 0.5
    beta2 = 0.9999
    amsgrad = False

    if 'decay' in configs:
        decay = configs['decay']
    if 'amsgrad' in configs:
        amsgrad = configs['amsgrad']
    if ('beta1' in configs) and ('beta2' in configs):
        beta1 = configs['beta1']
        beta2 = configs['beta2']

    if configs['optim'] == 'Adam':

        optim_d = torch.optim.Adam(Disc.parameters(), lr=configs[
                                   'lr_d'], betas=(beta1, beta2), weight_decay=decay)
        optim_g = torch.optim.Adam(gen, configs['lr_g'], betas=(
            beta1, beta2), weight_decay=decay)

    elif configs['optim'] == 'OAdam':

        optim_d = OAdam(Disc.parameters(), lr=configs['lr_d'], betas=(
            beta1, beta2), weight_decay=decay, amsgrad=amsgrad)
        optim_g = OAdam(gen, configs['lr_g'], betas=(
            beta1, beta2), weight_decay=decay, amsgrad=amsgrad)

    elif configs['optim'] == 'OMD':

        optim_d = OptMirrorAdam(Disc.parameters(), lr=configs['lr_d'], betas=(
            beta1, beta2), weight_decay=decay, amsgrad=amsgrad)
        optim_g = OptMirrorAdam(gen, configs['lr_g'], betas=(
            beta1, beta2), weight_decay=decay, amsgrad=amsgrad)

    train_loader = DataLoader(data, batch_size=BS, shuffle=True)

    return Gx1, Gx2, Gz1, Gz2, Disc, optim_g, optim_d, train_loader, cuda, configs


def train_epoch_hali(Gz1, Gz2, Gx1, Gx2, Disc, optim_d, optim_g, loader, epoch, cuda, configs):
    """
    :param Gz1: Level 1 of Encoder
    :param Gz2: Level 2 of Encoder
    :param Gx1: Level 1 of Decoder/Generator
    :param Gx2: Level 2 of Decoder/Generator
    :param Disc: Discriminator
    :param optim_d: Optimizer for Discriminator params
    :param optim_g: Optimizer for Generator params
    :param loader: data loader
    :param epoch: number of current epoch
    :param cuda: whether we are running on cuda
    :param configs: a dictionary with all params necessary for training
    :return g_loss: Generator loss
    :return d_loss: Decoder loss
    :return d_true: Prediction for x
    :return d_false: Prediction for x_hat
    """

    ncritic = configs['n_critic']
    cnt = 0
    gcnt = 0
    df = 0
    dt = 0
    dl = 0
    gl = 0

    for i, (imgs) in enumerate(loader):

        loss_d = runloop_d_hali(imgs, Gx1, Gx2, Gz1, Gz2,
                                Disc, optim_d, cuda, configs)

        if i % ncritic == 0:

            if 'unrolled_steps' in configs and configs['unrolled_steps'] > 1:
                loss_g, d_true, d_fake, Disc = runloop_g_hali_unrolled(
                    imgs, Gx1, Gx2, Gz1, Gz2, Disc, optim_g, optim_d, cuda, configs, loader)
            else:
                loss_g, d_true, d_fake = runloop_g_hali(
                    imgs, Gx1, Gx2, Gz1, Gz2, Disc, optim_g, cuda, configs)

            gl = gl + loss_g
            df = df + d_fake.item()
            dt = dt + d_true.data.mean().item()
            gcnt = gcnt + 1

        cnt = cnt + 1
        dl = dl + loss_d

    g_loss = gl / gcnt
    d_loss = dl / cnt

    d_true = dt / gcnt
    d_false = df / gcnt

    return g_loss, d_loss, d_true, d_false


def training_loop_hali(Gz1, Gz2, Gx1, Gx2, Disc, optim_d, optim_g, train_loader, configs, experiment, cuda):
    """
    :param Gz1: Level 1 of Encoder
    :param Gz2: Level 2 of Encoder
    :param Gx1: Level 1 of Decoder/Generator
    :param Gx2: Level 2 of Decoder/Generator
    :param Disc: Discriminator
    :param optim_d: Optimizer for Discriminator params
    :param optim_g: Optimizer for Generator params
    :param train_loader: data loader
    :param configs: a dictionary with all params necessary for training
    :param experiment: comet_ml experiment variable to store results
    :param cuda: whether we are running on cuda
    """

    Zdim = configs['Zdim']
    if 'continue_from' in configs:
        if configs['continue_from'] == -1:
            start_epoch = get_max_epoch(configs) - 1
            end_epoch = start_epoch + configs['n_epochs']
        else:
            start_epoch = configs['continue_from']
            end_epoch = start_epoch + configs['n_epochs']
    else:
        start_epoch = 0
        end_epoch = configs['n_epochs']

    if 'lr_scheduler' in configs:
        if configs['lr_scheduler'] == 'cyclic':
            milestones = [int(np.floor((2**x) * 1.2)) for x in range(7)]
            if configs['scheduler_mode'] == 'both':
                scheduler1 = CyclicCosAnnealingLR(
                    optim_d, milestones=milestones, eta_min=1e-5)
                scheduler2 = CyclicCosAnnealingLR(
                    optim_g, milestones=milestones, eta_min=1e-5)
            else:
                scheduler1 = CyclicCosAnnealingLR(
                    optim_d, milestones=milestones, eta_min=1e-5)

        if 'continue_from' in configs and configs['continue_from'] == -1:
            optim_d.load_state_dict(torch.load(
                configs['MODEL_PATH'], 'optim_d.pth'))
            optim_g.load_state_dict(torch.load(
                configs['MODEL_PATH'], 'optim_g.pth'))
            scheduler1.load_state_dict(torch.load(
                os.path.join(configs['MODEL_PATH'], 'scheduler1.pth')))
            if configs['scheduler_mode'] == 'both':
                scheduler1.load_state_dict(torch.load(
                    os.path.join(configs['MODEL_PATH'], 'scheduler2.pth')))

    for epoch in range(start_epoch, end_epoch):

        if 'lr_scheduler' in configs:
            scheduler1.step()
            if configs['scheduler_mode'] == 'both':
                scheduler2.step()

        g_loss, d_loss, d_true, d_false = train_epoch_hali(
            Gz1, Gz2, Gx1, Gx2, Disc, optim_d, optim_g, train_loader, epoch, cuda, configs
        )

        if 'lr_scheduler' in configs:
            torch.save(scheduler1.state_dict(), os.path.join(
                configs['MODEL_PATH'], 'scheduler1.pth'))
            if configs['scheduler_mode'] == 'both':
                torch.save(scheduler1.state_dict(), os.path.join(
                    configs['MODEL_PATH'], 'scheduler2.pth'))
        torch.save(optim_d.state_dict(), os.path.join(
            configs['MODEL_PATH'], 'optim_d.pth'))
        torch.save(optim_g.state_dict(), os.path.join(
            configs['MODEL_PATH'], 'optim_g.pth'))

        save_models_hali(Gz1, Gz2, Gx1, Gx2, Disc,
                         configs['MODEL_PATH'], epoch)
        sys.stdout.write("\r[%5d / %5d]: G: %.4f D: %.4f D(x,Gz(x)): %.4f D(Gx(z),z): %.4f" %
                         (epoch, configs['n_epochs'], g_loss, d_loss, d_true, d_false))

        experiment.log_metric('g_loss', g_loss)
        experiment.log_metric('d_loss', d_loss)
        experiment.log_metric('d_true', d_true)
        experiment.log_metric('d_fake', d_false)

        print()


def runloop_g_hali(imgs, Gx1, Gx2, Gz1, Gz2, Disc, optim_g, cuda, configs):
    """
    :param imgs: data for generator loop
    :param Gx1: Level 1 of Decoder/Generator
    :param Gx2: Level 2 of Decoder/Generator
    :param Gz1: Level 1 of Encoder
    :param Gz2: Level 2 of Encoder
    :param Disc: Discriminator
    :param optim_g: Optimizer for Generator params
    :param cuda: whether we are running on cuda
    :param configs: a dictionary with all params necessary for training
    :return loss_g: Generator loss
    :return d_false: Prediction for x_hat
    :return d_true: Prediction for x
    """

    softplus = nn.Softplus()
    Zdim = configs['Zdim']

    batch_size = imgs.size(0)

    if cuda:
        imgs = imgs.cuda()
    imgs = Variable(imgs)
    batch_size = imgs.size(0)

    z = torch.FloatTensor(batch_size, Zdim, 1, 1).normal_(0, 1)

    zv = Variable(z).cuda()

    encoded1 = Gz1(imgs)
    z1 = reparameterize(encoded1)

    encoded2 = Gz2(z1)
    z2 = reparameterize(encoded2)

    zv_enc = Gx1(zv)

    zv1 = reparameterize(zv_enc)

    imgs_fake = Gx2(zv1)

    def g_closure():

        Gx1.zero_grad()
        Gx1.zero_grad()

        Gz1.zero_grad()
        Gz2.zero_grad()

        d_true = Disc(imgs, z1, z2)
        d_fake = Disc(imgs_fake, zv1, zv)

        loss_g = torch.mean(softplus(d_true) + softplus(-d_fake))

        loss_g.backward(retain_graph=True)
        return loss_g.data.item(), d_fake.data.mean(), d_true.data.mean()
    loss_g, d_fake, d_true = optim_g.step(g_closure)
    return loss_g, d_true, d_fake


def runloop_d_hali(imgs, Gx1, Gx2, Gz1, Gz2, Disc, optim_d, cuda, configs):
    """
    :param imgs: data for generator loop
    :param Gx1: Level 1 of Decoder/Generator
    :param Gx2: Level 2 of Decoder/Generator
    :param Gz1: Level 1 of Encoder
    :param Gz2: Level 2 of Encoder
    :param Disc: Discriminator
    :param optim_d: Optimizer for Discriminator params
    :param cuda: whether we are running on cuda
    :param configs: a dictionary with all params necessary for training
    :return loss_d: Discriminator loss
    """

    softplus = nn.Softplus()
    Zdim = configs['Zdim']

    batch_size = imgs.size(0)

    if cuda:
        imgs = imgs.cuda()
    imgs = Variable(imgs)
    batch_size = imgs.size(0)

    z = torch.FloatTensor(batch_size, Zdim, 1, 1).normal_(0, 1).cuda()

    zv = Variable(z)

    encoded1 = Gz1(imgs)

    z1 = reparameterize(encoded1)

    encoded2 = Gz2(z1)
    z2 = reparameterize(encoded2)

    zv_enc = Gx1(zv)

    zv1 = reparameterize(zv_enc)

    imgs_fake = Gx2(zv1)

    def d_closure():

        Disc.zero_grad()
        batch_size = imgs.size(0)

        d_true = Disc(imgs, z1, z2)
        d_fake = Disc(imgs_fake, zv1, zv)

        if configs['gp']:

            gp = calc_gradient_penalty_hali(
                Disc, imgs, imgs_fake, zv1, z1, zv, z2, configs['gp_lambda'])
            loss_d = torch.mean(softplus(-d_true) + softplus(d_fake)) + gp

        else:
            loss_d = torch.mean(softplus(-d_true) + softplus(d_fake))

        loss_d.backward(retain_graph=True)
        return loss_d.data.item()

    loss_d = optim_d.step(d_closure)

    return loss_d


def runloop_g_hali_unrolled(imgs, Gx1, Gx2, Gz1, Gz2, Disc, optim_g, optim_d, cuda, configs, loader):
    """
    function runloop_g_hali_unrolled is a function that manages unrolled alternative training.
    Training in this way is significantly slower, but more stable

    :param imgs: data for generator loop
    :param Gx1: Level 1 of Decoder/Generator
    :param Gx2: Level 2 of Decoder/Generator
    :param Gz1: Level 1 of Encoder
    :param Gz2: Level 2 of Encoder
    :param Disc: Discriminator
    :param optim_g: Optimizer for Generator params
    :param cuda: whether we are running on cuda
    :param configs: a dictionary with all params necessary for training
    :return loss_g: Generator loss
    :return d_false: Prediction for x_hat
    :return d_true: Prediction for x
    """

    softplus = nn.Softplus()
    Zdim = configs['Zdim']

    backup_disc = copy.deepcopy(Disc.state_dict())
    for i in range(configs['unrolled_steps']):
        im = next(iter(loader))
        loss_d = runloop_d_hali(im, Gx1, Gx2, Gz1, Gz2,
                                Disc, optim_d, cuda, configs)

    loss_g, d_true, d_fake = runloop_g_hali(
        imgs, Gx1, Gx2, Gz1, Gz2, Disc, optim_g, cuda, configs)
    Disc.load_state_dict(backup_disc)
    del backup_disc
    return loss_g, d_true, d_fake, Disc


def get_max_epoch(configs):
    """
    get_max_epoch is a function that returns the highest epoch trainined so far for evaluation
    :param configs: a dictionary with all params necessary for training
    :returns highest epoch in saved models directory
    """

    onlyfiles = [f for f in listdir(configs['MODEL_PATH']) if isfile(
        join(configs['MODEL_PATH'], f))]

    epoch = []
    for s in onlyfiles:
        if ('scheduler' not in s) and ('optim' not in s):
            n = re.findall(r'\d+', s)
            epoch.append(int(n[len(n) - 1]))
    return(max(epoch))


def get_experiments():
    """
    get_experiments is a function that returns the names of all experiments in the experiment folder for evaluation especially in grid search scenario
    :returns list of experiments in the experiments directory
    """
    experiments = [f for f in listdir(
        'experiments') if isdir(join('experiments', f))]
    return experiments


def save_res_figure(configs, accuracies, f1_list):
    """
    save_res_figure is a function that saves plotted results to image directory during evaluation.
    it will fail on the cluster and is purely for convenience/local use.

    :param configs: a dictionary with all params necessary for training
    :param accuracies: a list of accuracy scores for all epochs
    :param f1_list: a list of f1 scores for all epochs
    """

    print()

    # commented to avoid "stack smashing detected" error on cluster
    # import matplotlib
    # import matplotlib.pyplot as plt
    # fig = plt.figure()
    # ax = plt.axes()

    # ax.plot(f1_list, label='F1 Score')
    # ax.plot(accuracies, label='Accuracy')
    # ax.legend(loc='best')
    # plt.title(configs['experiment'])
    # formatter = matplotlib.ticker.StrMethodFormatter("{x:.0f}")
    # plt.gca().xaxis.set_major_formatter(formatter)
    # plt.savefig(configs['IMAGE_PATH'] + '/clustering_results.png')


def get_results_ali(configs, experiment, train, labeled, valid_data):
    """
    get_results_ali is a function that evaluates classification performance by selecting the best of 5 linear SVMs at each training epoch.

    :param configs: a dictionary with all params necessary for training
    :param experiment: comet_ml experiment variable
    :param train: unlabeled training data
    :param labeled: labeled data
    :param valid_data: validation data
    :returns best_f1 : highest f1 score
    :returns best_accuracy: highest accuracy
    :returns best_model : best model epoch

    """

    Gx, Gz, Disc, optim_g, optim_d, train_loader, cuda, configs = initialize_ali(
        configs, train)

    max_ep = get_max_epoch(configs)
    best_accuracy = 0
    best_model = 0
    best_f1 = 0

    accuracies = []
    f1_scores = []
    for i in range(1, max_ep + 1):
        configs['continue_from'] = i
        Gx, Gz, Disc, optim_g, optim_d, train_loader, cuda, configs = initialize_ali(
            configs, train)
        train_labeled_enc, train_labels = get_ali_embeddings(Gz, labeled)
        valid_enc, val_labels = get_ali_embeddings(Gz, valid_data)
        svm = SVMClustering(configs['seed'])
        svm.train(train_labeled_enc, train_labels)
        y_pred = svm.predict_cluster(valid_enc)
        y_true = val_labels
        accuracy, f1 = compute_metrics(y_true, y_pred)
        accuracies.append(accuracy)
        f1_scores.append(f1)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_f1 = f1
            best_model = i

        experiment.log_metric('accuracy', accuracy)
        experiment.log_metric('f1_score', f1)
    save_res_figure(configs, accuracies, f1_scores)
    return(best_f1, best_accuracy, best_model)


def get_results_hali(configs, experiment, train, labeled, valid_data):
    """
    get_results_hali is a function that evaluates classification performance by selecting the best of 5 linear SVMs at each training epoch.

    :param configs: a dictionary with all params necessary for training
    :param experiment: comet_ml experiment variable
    :param train: unlabeled training data
    :param labeled: labeled data
    :param valid_data: validation data
    :returns best_f1 : highest f1 score
    :returns best_accuracy: highest accuracy
    :returns best_model : best model epoch

    """
    Gx1, Gx2, Gz1, Gz2, Disc, optim_g, optim_d, train_loader, cuda, configs = initialize_hali(
        configs, train)

    max_ep = get_max_epoch(configs)
    best_accuracy = 0
    best_model = 0
    best_f1 = 0
    accuracies = []
    f1_list = []

    for i in range(1, max_ep + 1):
        configs['continue_from'] = i
        Gx1, Gx2, Gz1, Gz2, Disc, optim_g, optim_d, train_loader, cuda, configs = initialize_hali(
            configs, train)
        train_labeled_enc, train_labels = get_hali_embeddings(
            Gz1, Gz2, labeled, 'z1')
        valid_enc, val_labels = get_hali_embeddings(Gz1, Gz2, valid_data, 'z1')
        save_recon_hali(Gx1, Gx2, Gz1, Gz2, i, True,
                        configs['IMAGE_PATH'], labeled)
        save_recon_hali(Gx1, Gx2, Gz1, Gz2, i, False,
                        configs['IMAGE_PATH'], labeled)
        acc_tmp = []
        f1_tmp = []
        for i in range(5):
            svm = SVMClustering(randint(0, 5000))
            svm.train(train_labeled_enc, train_labels)
            y_pred = svm.predict_cluster(valid_enc)
            y_true = val_labels
            accuracy, f1 = compute_metrics(y_true, y_pred)
            acc_tmp.append(accuracy)
            f1_tmp.append(f1)
        accuracies.append(max(acc_tmp))
        f1_list.append(max(f1_tmp))
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_f1 = f1
            best_model = i

        experiment.log_metric('accuracy', accuracy)
        experiment.log_metric('f1_score', f1)

    save_res_figure(configs, accuracies, f1_list)
    return(best_f1, best_accuracy, best_model)


def get_hali_embeddings(Gz1, Gz2, data, mode):
    """
    get_hali_embeddings is a function that returns embeddings from prescribed level of HALI's latent space

    :param Gz1: Gz1 model
    :param Gz2: Gz2 model
    :param data: the data to be used
    :param mode: which level of the latent space to use (z1, z2, or zcat)
    :returns all_embeddings : returns list of embedding vectors
    :returns all_targets: labels corresponding to embeddings list
    """

    all_embeddings = []
    all_targets = []
    loader = DataLoader(data, batch_size=32, shuffle=True)
    cuda = True if torch.cuda.is_available() else False
    labeled = True
    if loader.dataset.data.shape[0] > 500:
        labeled = False

    for imgs in loader:

        if labeled:
            (imgs, target) = imgs

        if cuda:
            data = Variable(imgs).cuda()
        else:
            data = Variable(imgs)
        encoded = Gz1(data)

        z1 = reparameterize(encoded)
        v1 = [z1.view(data.size()[0], -1).cpu().data.numpy()]

        enc_2 = Gz2(z1)
        z2 = reparameterize(enc_2)

        v2 = [z2.view(data.size()[0], -1).cpu().data.numpy()]

        vcat = np.concatenate([v1, v2], axis=2)

        if mode == 'cat':
            vec = vcat
        elif mode == 'z2':
            vec = v2
        else:
            vec = v1

        for l in range(np.shape(data)[0]):
            all_embeddings.append(vec[0][l, :])
            if labeled:
                all_targets.append(target[l].numpy()[0])

    return all_embeddings, all_targets


def get_ali_embeddings(Gz, data):
    """
    get_ali_embeddings is a function that returns embeddings from prescribed level of ALI's latent space

    :param Gz: Encoder
    :param data: the data to be used
    :returns all_embeddings : returns list of embedding vectors
    :returns all_targets: labels corresponding to embeddings list
    """

    all_embeddings = []
    all_targets = []
    loader = DataLoader(data, batch_size=32, shuffle=True)
    cuda = True if torch.cuda.is_available() else False
    labeled = True
    if loader.dataset.data.shape[0] > 500:
        labeled = False

    for imgs in loader:

        if labeled:
            (imgs, target) = imgs

        if cuda:
            data = Variable(imgs).cuda()
        else:
            data = Variable(imgs)

        encoded = Gz(data)

        z = reparameterize(encoded)
        v1 = [z.view(data.size()[0], -1).cpu().data.numpy()]

        for l in range(np.shape(v1)[1]):
            all_embeddings.append(v1[0][l, :])
            if labeled:
                all_targets.append(target[l].numpy()[0])

    return all_embeddings, all_targets


def saveimages_hali(Gx1, Gx2, noise1, noise2, IMAGE_PATH):
    """
    saveimages_hali is a function that saves generated images from both levels of the decoder/generator to the image path

    :param Gx1: Level 1 of Generator
    :param Gx2: Level 2 of Generator
    :param noise1: Noise variable of size of z2
    :param noise2: Noise variable of size of z1
    :param IMAGE_PATH: Image path
    """
    save_image(Gx2(noise2).cpu().data,
               os.path.join(IMAGE_PATH, '%d_1.png' % (epoch + 1)),
               nrow=9, padding=1,
               normalize=False)
    e1 = Gx2(reparameterize(Gx1(noise1)))
    save_image(e1.data,
               os.path.join(IMAGE_PATH, '%d_2.png' % (epoch + 1)),
               nrow=9, padding=1,
               normalize=False)


def save_recon_hali(Gx1, Gx2, Gz1, Gz2, epoch, from_z1, IMAGE_PATH, data):
    """
    save_recon_hali is a function that saves HALI reconstructions

    :param Gx1: Level 1 of Generator
    :param Gx2: Level 2 of Generator
    :param Gz1: Level 1 of Encoder
    :param Gz2: Level 2 of Encoder
    :param epoch: Number epoch
    :param IMAGE_PATH: Image path
    :param data: data to reproduce
    """

    dataloader = DataLoader(data, batch_size=32, shuffle=True)
    data = next(iter(dataloader))
    data = data[0]
    data = data.to('cuda')

    if not from_z1:
        latent = Gz1(data)  # z_hat

        z1 = reparameterize(latent)
        bbs = np.shape(data)[0]

        z_enc = Gz2(z1)

        recon = Gx2(reparameterize(Gx1(reparameterize(z_enc))))

        n = min(data.size(0), 8)

        ss = np.shape(data)
        comparison = torch.cat([data[:n],
                                recon.view(ss[0], ss[1], ss[2], ss[3])[:n]])
        save_image(comparison.cpu(),
                   IMAGE_PATH + '/reconstruction_z2_' + str(epoch) + '.png', nrow=n)
    else:

        latent = Gz1(data)  # z_hat

        z1 = reparameterize(latent)

        recon = Gx2(z1)

        n = min(data.size(0), 8)

        ss = np.shape(data)
        comparison = torch.cat([data[:n],
                                recon.view(ss[0], ss[1], ss[2], ss[3])[:n]])
        save_image(comparison.cpu(),
                   IMAGE_PATH + '/reconstruction_z1_' + str(epoch) + '.png', nrow=n)


def save_models_hali(Gz1, Gz2, Gx1, Gx2, Disc, MODEL_PATH, epoch):
    """
    save_recon_hali is a function that saves HALI reconstructions

    :param Gz1: Level 1 of Encoder
    :param Gz2: Level 2 of Encoder
    :param Gx1: Level 1 of Generator
    :param Gx2: Level 2 of Generator
    :param Disc: Discriminator
    :param MODEL_PATH: Models path
    :param epoch: Number epoch
    """

    torch.save(Gx1.state_dict(),
               os.path.join(MODEL_PATH, 'Gx1-%d.pth' % (epoch + 1)))
    torch.save(Gx2.state_dict(),
               os.path.join(MODEL_PATH, 'Gx2-%d.pth' % (epoch + 1)))
    torch.save(Gz1.state_dict(),
               os.path.join(MODEL_PATH, 'Gz1-%d.pth' % (epoch + 1)))
    torch.save(Gz2.state_dict(),
               os.path.join(MODEL_PATH, 'Gz2-%d.pth' % (epoch + 1)))
    torch.save(Disc.state_dict(),
               os.path.join(MODEL_PATH, 'Disc-%d.pth' % (epoch + 1)))


def save_models_ali(Gz, Gx, Disc, MODEL_PATH, epoch):
    """
    save_recon_ali is a function that saves ALI reconstructions

    :param Gz: Encoder
    :param Gx: Generator
    :param Disc: Discriminator
    :param MODEL_PATH: Models path
    :param epoch: Number epoch
    """
    torch.save(Gx.state_dict(),
               os.path.join(MODEL_PATH, 'Gx-%d.pth' % (epoch + 1)))
    torch.save(Gz.state_dict(),
               os.path.join(MODEL_PATH, 'Gz-%d.pth' % (epoch + 1)))
    torch.save(Disc.state_dict(),
               os.path.join(MODEL_PATH, 'Dict-%d.pth' % (epoch + 1)))


def calc_gradient_penalty_hali(discriminator, real_data, fake_data, z1, z_enc1, z2, z_enc2, gp_lambda):
    """Calculate Gradient Penalty HALI Variant 1
    Computes interpolates of all encodings before passing to the discriminator to account for gradients of encoder

    :param Disc: Discriminator
    :param real_data: real data
    :param fake_data: fake data
    :param z1: empirical z1
    :param z_enc1: encoded (fake) z1
    :param z2: empirical z2
    :param z_enc2: encoded (fake) z2
    :param gp_lambda: chosen lambda for gradient penalty
    :return gradient_penalty : the penalty
    """
    assert real_data.size(0) == fake_data.size(0)
    alpha = torch.rand(real_data.size(0), 1, 1, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.cuda()

    alpha_z1 = torch.rand(z1.size(0), 1, 1, 1)
    alpha_z1 = alpha_z1.expand(z1.size())
    alpha_z1 = alpha_z1.cuda()

    alpha_z2 = torch.rand(z2.size(0), 1, 1, 1)
    alpha_z2 = alpha_z2.expand(z2.size())
    alpha_z2 = alpha_z2.cuda()

    interpolates = Variable(alpha * real_data + ((1 - alpha) * fake_data),
                            requires_grad=True)

    interpolate_z1 = Variable(alpha_z1 * z_enc1 + ((1 - alpha_z1) * z1),
                              requires_grad=True)
    interpolate_z2 = Variable(alpha_z2 * z_enc2 + ((1 - alpha_z2) * z2),
                              requires_grad=True)

    disc_interpolates = discriminator(
        interpolates, interpolate_z1, interpolate_z2)
    gradients = torch.autograd.grad(
        outputs=disc_interpolates, inputs=interpolates,
        grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
        create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.contiguous().view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * gp_lambda

    return gradient_penalty


def calc_gradient_penalty_ali(discriminator, real_data, fake_data, z, z_enc,
                              gp_lambda):
    """Calculate Gradient Penalty HALI Variant 1
    Computes interpolates of all encodings before passing to the discriminator to account for gradients of encoder

    :param Disc: Discriminator
    :param real_data: real data
    :param fake_data: fake data
    :param z: empirical z
    :param z_enc: encoded (fake) z
    :return gradient_penalty : the penalty
    """
    assert real_data.size(0) == fake_data.size(0)
    alpha = torch.rand(real_data.size(0), 1, 1, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.cuda()

    alpha_z = torch.rand(z.size(0), 1, 1, 1)
    alpha_z = alpha_z.expand(z.size())
    alpha_z = alpha_z.cuda()

    interpolates = Variable(alpha * real_data + ((1 - alpha) * fake_data),
                            requires_grad=True)
    interpolate_z = Variable(alpha_z * z_enc + ((1 - alpha_z) * z),
                             requires_grad=True)
    disc_interpolates = discriminator(interpolates, interpolate_z)
    gradients = torch.autograd.grad(
        outputs=disc_interpolates, inputs=interpolates,
        grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
        create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.contiguous().view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * gp_lambda

    return gradient_penalty


def calc_gradient_penalty2_hali(discriminator, real_data, fake_data, encoder1, encoder2, gp_lambda):
    """Calculate Gradient Penalty HALI Variant 2.
    Passes interpolates through both encoders to the discriminator to account for encoding

    :param Disc: Discriminator
    :param real_data: real data
    :param fake_data: fake data
    :param encoder1: encoder for z1
    :param encoder2: encoder for z2
    :param gp_lambda: chosen lambda for gradient penalty
    :return gradient_penalty : the penalty
    """
    assert real_data.size(0) == fake_data.size(0)
    alpha = torch.rand(real_data.size(0), 1, 1, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.cuda()

    interpolates = Variable(alpha * real_data + ((1 - alpha) * fake_data),
                            requires_grad=True)
    enc1 = encoder1(interpolates)
    interpolate_z1 = reparameterize(enc1)

    enc2 = encoder2(interpolate_z1)
    interpolate_z2 = reparameterize(enc2)

    disc_interpolates = discriminator(
        interpolates, interpolate_z1.detach(), interpolate_z2.detach())
    gradients = torch.autograd.grad(
        outputs=disc_interpolates, inputs=interpolates,
        grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
        create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.contiguous().view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * gp_lambda

    return gradient_penalty


def calc_gradient_penalty2_ali(discriminator, real_data, fake_data, encoder, gp_lambda):
    """Calculate Gradient Penalty for ALI feeds interpolates through encoder to discriminator.
    :param Disc: Discriminator
    :param real_data: real data
    :param fake_data: fake data
    :param encoder: encoder
    :param gp_lambda: chosen lambda for gradient penalty
    :return gradient_penalty : the penalty
    """
    assert real_data.size(0) == fake_data.size(0)
    alpha = torch.rand(real_data.size(0), 1, 1, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.cuda()

    interpolates = Variable(alpha * real_data + ((1 - alpha) * fake_data),
                            requires_grad=True)

    z_enc = encoder(interpolates)

    interpolate_z = reparameterize(z_enc)

    disc_interpolates = discriminator(interpolates, interpolate_z.detach())
    gradients = torch.autograd.grad(
        outputs=disc_interpolates, inputs=interpolates,
        grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
        create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.contiguous().view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * gp_lambda

    return gradient_penalty


def reparameterize(encoded):
    """Reparameterization trick of:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    :param Disc: Discriminator
    :return reparameterized data
    """

    zd = encoded.size(1) // 2
    mu, logvar = encoded[:, :zd], encoded[:, zd:]
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return eps.mul(std).add_(mu)
