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

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

def initialize_ali(configs,data):


    IMAGE_PATH = configs['IMAGE_PATH']
    MODEL_PATH = configs['MODEL_PATH']

    if not os.path.exists(IMAGE_PATH):
        print('mkdir ', IMAGE_PATH)
        os.mkdir(IMAGE_PATH)
    if not os.path.exists(MODEL_PATH):
        print('mkdir ', MODEL_PATH)
        os.mkdir(MODEL_PATH)

    Zdim = configs['Zdim']
    BS   = configs['batch_size'] 

    Gx = ali.GeneratorX(zd=Zdim, ch=3)

    Gz = ali.GeneratorZ(zd=Zdim,ch = 3)


    Disc = ali.Discriminator(ch=3, zd= Zdim)

    if 'continue_from' in configs:

        Gx.load_state_dict(torch.load(MODEL_PATH+'/Gx-'+str(configs['continue_from'])+'.pth'))    
        Gz.load_state_dict(torch.load(MODEL_PATH+'/Gz-'+str(configs['continue_from'])+'.pth')) 
        Disc.load_state_dict(torch.load(MODEL_PATH+'/Dict-'+str(configs['continue_from'])+'.pth'))



    z_pred = torch.FloatTensor(81,Zdim,1,1).normal_(0,1)
   
    z_pred = Variable(z_pred)

    cuda = True if torch.cuda.is_available() else False


    if cuda:
        Gx.cuda()
        Gz.cuda()
        Disc.cuda()
        z_pred.cuda()
        
    gen = chain(Gx.parameters(),Gz.parameters())


    if configs['optim']=='Adam':

        optim_d = torch.optim.Adam(Disc.parameters(),lr=configs['lr_d'],betas=(0.5, .999),weight_decay=0)
        optim_g = torch.optim.Adam(gen,configs['lr_g'], betas=(0.5, .999), weight_decay=0)

    elif configs['optim']=='OAdam':

        optim_d = OAdam(Disc.parameters(),lr=configs['lr_d'],betas=(0.5, .999),weight_decay=0,amsgrad =configs['amsgrad'])
        optim_g = OAdam(gen,configs['lr_g'], betas=(0.5, .999), weight_decay=0,amsgrad =configs['amsgrad'])

    elif configs['optim']=='OMD':

        optim_d = OptMirrorAdam(Disc.parameters(),lr=configs['lr_d'],betas=(0.5, .999),weight_decay=0,amsgrad =configs['amsgrad'],extragradient=configs['extragradient'])
        optim_g = OptMirrorAdam(gen,configs['lr_g'], betas=(0.5, .999), weight_decay=0,amsgrad =configs['amsgrad'],extragradient=configs['extragradient'])
   
    train_loader = DataLoader(data, batch_size=BS, shuffle=True)

    return Gx,Gz,Disc,z_pred,optim_g,optim_d,train_loader,cuda



def train_epoch_ali(Gz,Gx,Disc, optim_d,optim_g, loader,epoch,cuda,configs):
    """
    Trains model for a single epoch
    :param model: the model created under src/algorithms
    :param optimizer: pytorch optim
    :param loader: the training set loader
    :param include_subsamples: whether to train the principal ode_network with sub samples of the signal
    
    :return: training loss, accuracy, large (for next epoch)
    """

    ncritic = configs['n_critic']
    cnt = 0
    gcnt = 0
    df = 0
    dt = 0
    dl = 0
    gl = 0

    for i, (imgs) in enumerate(loader):
     
        loss_d = runloop_d_ali(imgs,Gx,Gz,Disc,optim_d,cuda,configs)
        
        if i%ncritic==0 or not is_critic:
            loss_g,d_true,d_fake = runloop_g_ali(imgs,Gx,Gz,Disc,optim_g,cuda,configs) 

            gl = gl + loss_g
            df = df +d_fake.item()
            dt = dt + d_true.data.mean().item()
            gcnt = gcnt+1

        cnt = cnt+1
        dl = dl+loss_d
        
    g_loss = gl/gcnt
    d_loss = dl/cnt 

    d_true  = dt/gcnt
    d_false =  df/gcnt    

    # generate fake images

    # saveimages(Gx1,Gx2,Gz1,Gz2,z_pred1,z_pred2)
    # test(Gx1,Gx2,Gz1,Gz2,epoch,True)
    # test(Gx1,Gx2,Gz1,Gz2,epoch,False)
 


    return g_loss,d_loss,d_true,d_false
def training_loop_ali(Gz,Gx,Disc,optim_d,optim_g,train_loader,configs,experiment,cuda,z_pred):
    """
     Runs training loop
    :param model: the model created under src/algorithms
    :param optimizer: pytorch optim
    :param train_loader: the training set loader
    :param valid_loader: the validation set loader
    :param hyperparameters_dict: stores save location of model
    
    :return: [(train_losses,train_accuracy)(valid_losses,valid_accuracy)]
    """

    # Index starts at 1 for reporting purposes
    
    Zdim = configs['Zdim']
    if 'continue_from' in configs:
        start_epoch = configs['continue_from']+1
        end_epoch   = start_epoch + configs['n_epochs']
    else:
        start_epoch = 0
        end_epoch = configs['n_epochs']
    for epoch in range(start_epoch, end_epoch):

        g_loss,d_loss,d_true,d_false = train_epoch_ali(
            Gz,Gx,Disc, optim_d,optim_g, train_loader,epoch,cuda,configs
        )


        # saveimages_hali(Gx1,Gx2,Gz1,Gz2,z_pred1,z_pred2,configs['IMAGE_PATH'])
        # save_recon_hali(Gx1,Gx2,Gz1,Gz2,epoch,True,configs['IMAGE_PATH'])
        # save_recon_hali(Gx1,Gx2,Gz1,Gz2,epoch,False,configs['IMAGE_PATH'])

        save_models_ali(Gz,Gx,Disc,configs['MODEL_PATH'],epoch)
        sys.stdout.write("\r[%5d / %5d]: G: %.4f D: %.4f D(x,Gz(x)): %.4f D(Gx(z),z): %.4f" % (epoch,configs['n_epochs'],g_loss,d_loss,d_true,d_false))
        
        experiment.log_metric('g_loss', g_loss)
        experiment.log_metric('d_loss', d_loss)
        experiment.log_metric('d_true', d_true)
        experiment.log_metric('d_fake', d_false)

        print()     

def runloop_g_ali(imgs,Gx,Gz,Disc,optim_g,cuda,configs):
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

        d_true = Disc(imgs,z)
        d_fake = Disc(imgs_fake,zv)
        
        loss_g = torch.mean(softplus(d_true) + softplus(-d_fake))

        loss_g.backward(retain_graph=True)
        return loss_g.data.item(),d_fake.data.mean(),d_true.data.mean()
    loss_g,d_fake,d_true = optim_g.step(g_closure)
    return loss_g, d_true,d_fake

def runloop_d_ali(imgs,Gx,Gz,Disc,optim_d,cuda,configs):
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

        d_true = Disc(imgs,z)
        d_fake = Disc(imgs_fake,zv)

        gp = calc_gradient_penalty2_ali(Disc,imgs, imgs_fake, zv,z)
        
        loss_d = torch.mean(softplus(-d_true) + softplus(d_fake))+gp

        loss_d.backward(retain_graph=True)
        return loss_d.data.item()

    loss_d = optim_d.step(d_closure)

    return loss_d



def initialize_hali(configs,data):


    IMAGE_PATH = configs['IMAGE_PATH']
    MODEL_PATH = configs['MODEL_PATH']

    if not os.path.exists(IMAGE_PATH):
        print('mkdir ', IMAGE_PATH)
        os.mkdir(IMAGE_PATH)
    if not os.path.exists(MODEL_PATH):
        print('mkdir ', MODEL_PATH)
        os.mkdir(MODEL_PATH)

    Zdim = configs['Zdim']
    zd1  = configs['z1dim']
    BS   = configs['batch_size'] 

    Gx1 = hali.GeneratorX1(zd=Zdim, ch=3,zd1 = zd1)
    Gx2 = hali.GeneratorX2(zd=Zdim, ch=3,zd1=zd1)

    Gz1 = hali.GeneratorZ1(zd=Zdim,ch = 3,zd1 = zd1)
    Gz2 = hali.GeneratorZ2(zd=Zdim, zd1=zd1)

    Disc = hali.Discriminator(ch=3, zd= Zdim,zd1 = zd1)

    if 'continue_from' in configs:

        Gx1.load_state_dict(torch.load(MODEL_PATH+'/Gx1-'+str(configs['continue_from'])+'.pth')) 
        Gx2.load_state_dict(torch.load(MODEL_PATH+'/Gx2-'+str(configs['continue_from'])+'.pth')) 
        Gz1.load_state_dict(torch.load(MODEL_PATH+'/Gz1-'+str(configs['continue_from'])+'.pth')) 
        Gz2.load_state_dict(torch.load(MODEL_PATH+'/Gz2-'+str(configs['continue_from'])+'.pth'))
        Disc.load_state_dict(torch.load(MODEL_PATH+'/Disc-'+str(configs['continue_from'])+'.pth'))

    z_pred1 = torch.FloatTensor(81,Zdim,1,1).normal_(0,1)
    z_pred2 = torch.FloatTensor(81, zd1, 16, 16).normal_(0, 1)
    z_pred2 = Variable(z_pred2)
    z_pred1 = Variable(z_pred1)

    cuda = True if torch.cuda.is_available() else False


    if cuda:
        Gx1.cuda()
        Gz1.cuda()
        Gx2.cuda()
        Gz2.cuda()
        Disc.cuda()
        z_pred1.cuda()
        z_pred2.cuda()

    gen = chain(Gx1.parameters(),Gx2.parameters(),Gz1.parameters(),Gz2.parameters())


    if configs['optim']=='Adam':

        optim_d = torch.optim.Adam(Disc.parameters(),lr=configs['lr_d'],betas=(0.5, .999),weight_decay=0)
        optim_g = torch.optim.Adam(gen,configs['lr_g'], betas=(0.5, .999), weight_decay=0)

    elif configs['optim']=='OAdam':

        optim_d = OAdam(Disc.parameters(),lr=configs['lr_d'],betas=(0.5, .999),weight_decay=0,amsgrad =configs['amsgrad'])
        optim_g = OAdam(gen,configs['lr_g'], betas=(0.5, .999), weight_decay=0,amsgrad =configs['amsgrad'])

    elif configs['optim']=='OMD':

        optim_d = OptMirrorAdam(Disc.parameters(),lr=configs['lr_d'],betas=(0.5, .999),weight_decay=0,amsgrad =configs['amsgrad'])
        optim_g = OptMirrorAdam(gen,configs['lr_g'], betas=(0.5, .999), weight_decay=0,amsgrad =configs['amsgrad'])
   
    train_loader = DataLoader(data, batch_size=BS, shuffle=True)

    return Gx1,Gx2,Gz1,Gz2,Disc,z_pred1,z_pred2,optim_g,optim_d,train_loader,cuda

def runloop_g_hali(imgs,Gx1,Gx2,Gz1,Gz2,Disc,optim_g,cuda,configs):
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


        d_true = Disc(imgs,z1,z2)
        d_fake = Disc(imgs_fake,zv1,zv)
        

        loss_g = torch.mean(softplus(d_true) + softplus(-d_fake))

        loss_g.backward(retain_graph=True)
        return loss_g.data.item(),d_fake.data.mean(),d_true.data.mean()
    loss_g,d_fake,d_true = optim_g.step(g_closure)
    return loss_g, d_true,d_fake

def runloop_d_hali(imgs,Gx1,Gx2,Gz1,Gz2,Disc,optim_d,cuda,configs):
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

    def d_closure():

        Disc.zero_grad()
        batch_size = imgs.size(0)

        d_true = Disc(imgs,z1,z2)
        d_fake = Disc(imgs_fake,zv1,zv)
        gp = calc_gradient_penalty2_hali(Disc,imgs, imgs_fake, zv1,z1,zv, z2,1)
        
        loss_d = torch.mean(softplus(-d_true) + softplus(d_fake))+gp

        loss_d.backward(retain_graph=True)
        return loss_d.data.item()

    loss_d = optim_d.step(d_closure)

    return loss_d


    
def train_epoch_hali(Gz1,Gz2,Gx1,Gx2,Disc, optim_d,optim_g, loader,epoch,cuda,configs):
    """
    Trains model for a single epoch
    :param model: the model created under src/algorithms
    :param optimizer: pytorch optim
    :param loader: the training set loader
    :param include_subsamples: whether to train the principal ode_network with sub samples of the signal
    
    :return: training loss, accuracy, large (for next epoch)
    """

    ncritic = configs['n_critic']
    cnt = 0
    gcnt = 0
    df = 0
    dt = 0
    dl = 0
    gl = 0

    for i, (imgs) in enumerate(loader):
     
        loss_d = runloop_d_hali(imgs,Gx1,Gx2,Gz1,Gz2,Disc,optim_d,cuda,configs)
        
        if i%ncritic==0 or not is_critic:
            loss_g,d_true,d_fake = runloop_g_hali(imgs,Gx1,Gx2,Gz1,Gz2,Disc,optim_g,cuda,configs) 

            gl = gl + loss_g
            df = df +d_fake.item()
            dt = dt + d_true.data.mean().item()
            gcnt = gcnt+1

        cnt = cnt+1
        dl = dl+loss_d
        
    g_loss = gl/gcnt
    d_loss = dl/cnt 

    d_true  = dt/gcnt
    d_false =  df/gcnt    

    # generate fake images

    # saveimages(Gx1,Gx2,Gz1,Gz2,z_pred1,z_pred2)
    # test(Gx1,Gx2,Gz1,Gz2,epoch,True)
    # test(Gx1,Gx2,Gz1,Gz2,epoch,False)
 


    return g_loss,d_loss,d_true,d_false



def training_loop_hali(Gz1,Gz2,Gx1,Gx2,Disc,optim_d,optim_g,train_loader,configs,experiment,cuda,z_pred1,z_pred2):
    """
     Runs training loop
    :param model: the model created under src/algorithms
    :param optimizer: pytorch optim
    :param train_loader: the training set loader
    :param valid_loader: the validation set loader
    :param hyperparameters_dict: stores save location of model
    
    :return: [(train_losses,train_accuracy)(valid_losses,valid_accuracy)]
    """

    # Index starts at 1 for reporting purposes
    
    Zdim = configs['Zdim']
    if 'continue_from' in configs:
        start_epoch = configs['continue_from']+1
        end_epoch   = start_epoch + configs['n_epochs']
    else:
        start_epoch = 0
        end_epoch = configs['n_epochs']
    for epoch in range(start_epoch, end_epoch):

        g_loss,d_loss,d_true,d_false = train_epoch_hali(
            Gz1,Gz2,Gx1,Gx2,Disc, optim_d,optim_g, train_loader,epoch,cuda,configs
        )


        # saveimages_hali(Gx1,Gx2,Gz1,Gz2,z_pred1,z_pred2,configs['IMAGE_PATH'])
        # save_recon_hali(Gx1,Gx2,Gz1,Gz2,epoch,True,configs['IMAGE_PATH'])
        # save_recon_hali(Gx1,Gx2,Gz1,Gz2,epoch,False,configs['IMAGE_PATH'])

        save_models_hali(Gz1,Gz2,Gx1,Gx2,Disc,configs['MODEL_PATH'],epoch)
        sys.stdout.write("\r[%5d / %5d]: G: %.4f D: %.4f D(x,Gz(x)): %.4f D(Gx(z),z): %.4f" % (epoch,configs['n_epochs'],g_loss,d_loss,d_true,d_false))
        
        experiment.log_metric('g_loss', g_loss)
        experiment.log_metric('d_loss', d_loss)
        experiment.log_metric('d_true', d_true)
        experiment.log_metric('d_fake', d_false)

        print()     


def get_hali_embeddings(Gz1,Gz2,data,mode):
    all_embeddings=[]
    all_targets = []
    loader = DataLoader(data, batch_size = 32,shuffle=True)
    cuda = True if torch.cuda.is_available() else False
    labeled = True
    if loader.dataset.data.shape[0] >500 :
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

        vcat = np.concatenate([v1,v2],axis=2)

        if mode =='cat':
            vec = vcat
        elif mode =='z2':
            vec = v2
        else:
            vec = v1

        for l in range(np.shape(vcat)[1]):
            all_embeddings.append(vcat[0][l,:])
            if labeled:
                all_targets.append(target[l].numpy()[0])

    return all_embeddings,all_targets

def get_ali_embeddings(Gz,data):
    all_embeddings=[]
    all_targets = []
    loader = DataLoader(data, batch_size = 32,shuffle=True)
    cuda = True if torch.cuda.is_available() else False
    labeled = True
    if loader.dataset.data.shape[0] >500 :
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
            all_embeddings.append(v1[0][l,:])
            if labeled:
                all_targets.append(target[l].numpy()[0])

    return all_embeddings,all_targets


          
def calc_gradient_penalty_hali(discriminator, real_data, fake_data,encoder1,encoder2, gp_lambda):
    """Calculate Gradient Penalty HALI Variant 1."""
    """Passes interpolates through both encoders to the discriminator to account for encoding"""
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

    
    disc_interpolates = discriminator(interpolates, interpolate_z1.detach(),interpolate_z2.detach())
    gradients = torch.autograd.grad(
        outputs=disc_interpolates, inputs=interpolates,
        grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
        create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.contiguous().view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * gp_lambda

    return gradient_penalty

def calc_gradient_penalty2_hali(discriminator,real_data, fake_data, z1,z_enc1,z2, z_enc2,gp_lambda):
    """Calculate Gradient Penalty HALI Variant 2."""
    """Computes interpolates of all encodings before passing to the discriminator to account for gradients of encoder"""
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
    
    
    disc_interpolates = discriminator(interpolates, interpolate_z1,interpolate_z2)
    gradients = torch.autograd.grad(
        outputs=disc_interpolates, inputs=interpolates,
        grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
        create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.contiguous().view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * gp_lambda

    return gradient_penalty


def calc_gradient_penalty_ali(discriminator, real_data, fake_data, encoder, gp_lambda):
    """Calculate Gradient Penalty for ALI feeds interpolates through encoder to discriminator."""
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

def calc_gradient_penalty2_ali(discriminator,  real_data, fake_data, z, z_enc,
                          gp_lambda):
    """Calculate GP."""
    Disc,imgs, imgs_fake, zv
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
    interpolates_cond = Variable(fake_data_cond, requires_grad=True)
    disc_interpolates = discriminator(interpolates, interpolates_cond, interpolate_z)
    gradients = torch.autograd.grad(
        outputs=disc_interpolates, inputs=interpolates,
        grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
        create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.contiguous().view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * gp_lambda

    return gradient_penalty

def saveimages_hali(Gx1,Gx2,Gz1,Gz2,noise1,noise2,IMAGE_PATH):
    """Save Samples from HALI Latent spaces z1 and z2"""
    save_image(Gx2(noise2).cpu().data,
               os.path.join(IMAGE_PATH,'%d_1.png' % (epoch+1)),
               nrow=9, padding=1,
               normalize=False)
    e1 = Gx2(reparameterize(Gx1(noise1)))
    save_image(e1.data,
             os.path.join(IMAGE_PATH,'%d_2.png' % (epoch+1)),
             nrow=9, padding=1,
             normalize=False)

def save_recon_hali(Gx1,Gx2,Gz1,Gz2,epoch,from_z1,IMAGE_PATH):

        data=  next(iter(dataloader))
        data = data[0]
        data=data.to('cuda')

        if not from_z1:
            latent = Gz1(data)  #z_hat

            z1 = reparameterize(latent)
            bbs = np.shape(data)[0]

            z_enc = Gz2(z1)


            recon = Gx2(reparameterize(Gx1(reparameterize(z_enc))))



            n = min(data.size(0), 8)

            ss = np.shape(data)
            comparison = torch.cat([data[:n],
                                  recon.view(ss[0],ss[1],ss[2],ss[3])[:n]])
            save_image(comparison.cpu(),
                     IMAGE_PATH+'/reconstruction_z2_' + str(epoch) + '.png', nrow=n)
        else:

            latent = Gz1(data)  #z_hat

            z1 = reparameterize(latent)


            recon = Gx2(z1)



            n = min(data.size(0), 8)

            ss = np.shape(data)
            comparison = torch.cat([data[:n],
                                  recon.view(ss[0],ss[1],ss[2],ss[3])[:n]])
            save_image(comparison.cpu(),
                     IMAGE_PATH+'/reconstruction_z1_' + str(epoch) + '.png', nrow=n)


def save_models_hali(Gz1,Gz2,Gx1,Gx2,Disc,MODEL_PATH,epoch):
    # save models
    torch.save(Gx1.state_dict(),
               os.path.join(MODEL_PATH, 'Gx1-%d.pth' % (epoch+1)))
    torch.save(Gx2.state_dict(),
               os.path.join(MODEL_PATH, 'Gx2-%d.pth' % (epoch+1)))
    torch.save(Gz1.state_dict(),
               os.path.join(MODEL_PATH, 'Gz1-%d.pth' % (epoch+1)))
    torch.save(Gz2.state_dict(),
               os.path.join(MODEL_PATH, 'Gz2-%d.pth'  % (epoch+1)))
    torch.save(Disc.state_dict(),
               os.path.join(MODEL_PATH, 'Disc-%d.pth'  % (epoch+1)))

def save_models_ali(Gz,Gx,Disc,MODEL_PATH,epoch):
    torch.save(Gx.state_dict(),
               os.path.join(MODEL_PATH, 'Gx-%d.pth' % (epoch+1)))
    torch.save(Gz.state_dict(),
               os.path.join(MODEL_PATH, 'Gz-%d.pth' % (epoch+1)))
    torch.save(Disc.state_dict(),
               os.path.join(MODEL_PATH, 'Dict-%d.pth'  % (epoch+1)))
 



def reparameterize(encoded):

    """Reparameterization trick of:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)"""

    zd = encoded.size(1)//2
    mu,logvar = encoded[:, :zd], encoded[:, zd:]
    std = torch.exp(0.5*logvar)
    eps = torch.randn_like(std)
    return eps.mul(std).add_(mu)




