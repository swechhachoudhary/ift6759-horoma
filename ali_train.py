#!/usr/bin/env python
# coding: utf-8
# set params
# ===============
import numpy as np
import os
gpu = 0
cuda = 0 if gpu == -1 else 1
if cuda:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
BS = 6
Zdim =256
IMAGE_PATH = 'images_ali'
MODEL_PATH = 'models_ali'
ld = 1
is_gradient_penalty = True
N_Channels = 3
critic = 1
unrolled_steps = 10
num_epochs =100


# ===============
import torch
import copy
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable
from itertools import chain
from torchvision.utils import save_image

from torchvision.datasets import ImageFolder
if not os.path.exists(IMAGE_PATH):
    print('mkdir ', IMAGE_PATH)
    os.mkdir(IMAGE_PATH)
if not os.path.exists(MODEL_PATH):
    print('mkdir ', MODEL_PATH)
    os.mkdir(MODEL_PATH)

# -*- coding: utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import autograd

# -*- coding: utf-8 -*-
import sys




# load dataset
# ==========================


def runloop(imgs,Gx,Gz,Disc,optim_d,optim_g):
    batch_size = imgs.size(0)

    if cuda:
        imgs = imgs.cuda()
    imgs = Variable(imgs)
    batch_size = imgs.size(0)


    z = torch.FloatTensor(batch_size, Zdim, 1, 1).normal_(0, 1)

    zv = Variable(z).cuda()

  
    encoded1 = Gz(imgs)

    z_enc = reparameterize(encoded1)

    imgs_fake = Gx(zv)
    
    def g_closure():
        
        Gx.zero_grad()

        Gz.zero_grad()
        

        d_true = Disc(imgs,z_enc)
        d_fake = Disc(imgs_fake,zv)
        

        loss_g = torch.mean(softplus(d_true) + softplus(-d_fake))

        loss_g.backward(retain_graph=True)
        return loss_g.data.item(),d_fake.data.mean(),d_true.data.mean()
    loss_g,d_fake,d_true = optim_g.step(g_closure)

    def d_closure():

        Disc.zero_grad()

        d_true = Disc(imgs,z_enc)
        d_fake = Disc(imgs_fake,zv)
        gp = calc_gradient_penalty(Disc,imgs, imgs_fake, Gz,10)
        loss_d = torch.mean(softplus(-d_true) + softplus(d_fake))+gp

        loss_d.backward(retain_graph=True)
        return loss_d.data.item()
    loss_d = optim_d.step(d_closure)
    return loss_d,loss_g,d_true,d_fake



Gx = GeneratorX(zd=Zdim, ch=3)
Gz = GeneratorZ(zd=Zdim,ch = 3)
Disc = Discriminator(zd=Zdim,ch=3)
if cuda:
    Gx.cuda()
    Gz.cuda()
    Disc.cuda()
    z, z_pred, noise = z.cuda(), z_pred.cuda(), noise.cuda()

ld = 10
num_epochs = 200
global backup_dx;
global backup_dxz;
last_d = 0
last_g = 0
eps = 1e-15 # to avoid possible numerical instabilities during backward
unrolled_steps_cur = unrolled_steps
critic = 2
N = len(labelled_loader)
# train
# ==========================
softplus = nn.Softplus()
for epoch in range(num_epochs):
    cnt = 0
    df = 0
    dt = 0
    dl = 0
    gl = 0
    for i, (imgs,l) in enumerate(labelled_loader):

        loss_d,loss_g,d_true,d_fake = runloop(imgs,Gx,Gz,Disc,optim_d,optim_g)
        dl = dl + loss_d
        gl = gl + loss_g
        df = df +d_fake.item()
        dt = dt + d_true.data.mean().item()
        cnt = cnt+1
        prog_ali_reg(epoch, i+1, N, gl/cnt, dl/cnt, dt/cnt, df/cnt)

        if i%1000==0:
          save_image(Gx(z_pred).data,
          os.path.join(IMAGE_PATH,'%d.png' % (epoch+1)),
          nrow=9, padding=1,
          normalize=False)
          test(Gx,Gz,epoch,unlabelled_loader)
    # generate fake images
    save_image(Gx(z_pred).data,
               os.path.join(IMAGE_PATH,'%d.png' % (epoch+1)),
               nrow=9, padding=1,
               normalize=False)
    test(Gx,Gz,epoch,unlabelled_loader)

    # save models
    torch.save(Gx.state_dict(),
               os.path.join(MODEL_PATH, 'Gx-%d.pth' % (epoch+1)))
    torch.save(Gz.state_dict(),
               os.path.join(MODEL_PATH, 'Gz-%d.pth' % (epoch+1)))
    torch.save(Disc.state_dict(),
               os.path.join(MODEL_PATH, 'Dict-%d.pth'  % (epoch+1)))
 
    print()
