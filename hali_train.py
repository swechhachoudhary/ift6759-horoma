eps = 1e-12

ld  = 10
num_epochs = 10000
global backup_disc;

last_d = 0
last_g = 0 # to avoid possible numerical instabilities during backward
unrolled_steps_cur = 0
ncritic = 1
is_critic = True
Z1dim = 64
d_turn = True
g_turn = 1
def runloop_g(imgs,Gx1,Gx2,Gz1,Gz2,Disc,optim_g):
    batch_size = imgs.size(0)

    if cuda:
        imgs = imgs.cuda()
    imgs = Variable(imgs)
    batch_size = imgs.size(0)


    z = torch.FloatTensor(batch_size, Zdim, 1, 1).normal_(0, 1)

    zv = Variable(z).cuda()

    z1 = torch.FloatTensor(batch_size, Z1dim, 32, 32).normal_(0, 1)

    z1v = Variable(z1).cuda()

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
def runloop_d(imgs,Gx1,Gx2,Gz1,Gz2,Disc,optim_d):
    batch_size = imgs.size(0)

    if cuda:
        imgs = imgs.cuda()
    imgs = Variable(imgs)
    batch_size = imgs.size(0)


    z = torch.FloatTensor(batch_size, Zdim, 1, 1).normal_(0, 1)

    zv = Variable(z).cuda()

    z1 = torch.FloatTensor(batch_size, Z1dim, 32, 32).normal_(0, 1)

    z1v = Variable(z1).cuda()

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
        gp = calc_gradient_penalty2(Disc,imgs, imgs_fake, zv1,z1,zv, z2,1)
        
        loss_d = torch.mean(softplus(-d_true) + softplus(d_fake))+gp

        loss_d.backward(retain_graph=True)
        return loss_d.data.item()
    loss_d = optim_d.step(d_closure)
    return loss_d
    
# train
# ==========================
softplus = nn.Softplus()
for epoch in range(num_epochs):
    cnt = 0
    gcnt = 0
    df = 0
    dt = 0
    dl = 0
    gl = 0
    for i, (imgs, _) in enumerate(dataloader):
        
     
        loss_d = runloop_d(imgs,Gx1,Gx2,Gz1,Gz2,Disc,optim_d)
        
        if i%ncritic==0 or not is_critic:
            loss_g,d_true,d_fake = runloop_g(imgs,Gx1,Gx2,Gz1,Gz2,Disc,optim_g) 

            gl = gl + loss_g
            df = df +d_fake.item()
            dt = dt + d_true.data.mean().item()
            gcnt = gcnt+1
        cnt = cnt+1
        dl = dl+loss_d
 
        
        
        prog_ali_reg(epoch, i+1, N, gl/gcnt, dl/cnt, dt/gcnt, df/gcnt)
        torch.cuda.empty_cache()


    # generate fake images
    saveimages(Gx1,Gx2,Gz1,Gz2,z_pred1,z_pred2)
    test(Gx1,Gx2,Gz1,Gz2,epoch,True)
    test(Gx1,Gx2,Gz1,Gz2,epoch,False)
  
    # save models
    #     torch.save(Gx.state_dict(),
    #                os.path.join(MODEL_PATH, 'Gx-%d.pth' % (epoch+1)))
    #     torch.save(Gz.state_dict(),
    #                os.path.join(MODEL_PATH, 'Gz-%d.pth' % (epoch+1)))
    #     torch.save(Dx.state_dict(),
    #                os.path.join(MODEL_PATH, 'Dx-%d.pth'  % (epoch+1)))
    #     torch.save(Dxz.state_dict(),
    #                os.path.join(MODEL_PATH, 'Dxz-%d.pth'  % (epoch+1)))
    print()