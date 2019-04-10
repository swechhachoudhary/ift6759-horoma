class Discriminator(nn.Module):
    def __init__(self,ch=3,zd=256):
        super(Discriminator, self).__init__()
        self.Dx  = DiscriminatorX(zd=zd,ch=ch)
        self.Dxz = DiscriminatorXZ(zd=zd)
        
    def forward(self,x_input,z_input):   
        eps = 1e-12
        noise = Variable((torch.Tensor(x_input.size()).normal_(0, 0.1 * (num_epochs - epoch) / num_epochs))).cuda()
        
        dx_out = self.Dx(x_input+noise)
        noise = Variable((torch.Tensor(dx_out.size()).normal_(0, 0.1 * (num_epochs - epoch) / num_epochs))).cuda()
        
        d_out = self.Dxz(torch.cat((dx_out, z_input+noise), dim=1))+eps
       
        return d_out
      
class DiscriminatorX(nn.Module):
    def __init__(self, zd=128,ch=1):
        super().__init__()
        
        self.c1 = SpectralNorm(nn.Conv2d(ch, zd//4, 3, 2))
        self.lr = nn.LeakyReLU(0.02)

        self.c2 = SpectralNorm(nn.Conv2d(zd//4, zd//2, 3, 2))
            
        self.c3 =  SpectralNorm(nn.Conv2d(zd//2, zd, 3, 1))
            
        self.c4 =  SpectralNorm(nn.Conv2d(zd, zd, 3, 1))
     
        self.c5 =  SpectralNorm(nn.Conv2d(zd, zd, 3,1))
        
        self.d  = nn.Dropout2d(0.2)
            
    def forward(self, x):
        x = self.c1(x)
        x =self.lr(x)
        x = self.d(x)

        x = self.c2(x)
        x = self.lr(x)
        x = self.d(x)

        x = self.c3(x)
        x = self.lr(x)
        
        x = self.c4(x)
        x = self.lr(x)
        x = self.d(x)
        
        
        x = self.c5(x)
        x = self.lr(x)
        x = self.d(x)
        
        return x

class DiscriminatorXZ(nn.Module):
    def __init__(self, zd=128):
        super().__init__()
        self.net = nn.Sequential(
            SpectralNorm(nn.Conv2d(zd*2, zd*2, 1, 1)),
            nn.LeakyReLU(0.02),
            nn.Dropout2d(0.2),
            SpectralNorm(nn.Conv2d(zd*2, zd, 1, 1)),    
            nn.LeakyReLU(0.02),
            nn.Dropout2d(0.2),
            SpectralNorm(nn.Conv2d(zd, 1, 1, 1)),
        )

    def forward(self, x):
        return self.net(x)

class GeneratorZ(nn.Module):
    def __init__(self, zd=128, ch=1):
      super().__init__()
      self.c1 = nn.Conv2d(ch, zd//8, 3, 2)
      self.bn1 = nn.BatchNorm2d(zd//8)
      self.lr  = nn.LeakyReLU(0.02)

      self.c2  = nn.Conv2d(zd//8, zd//4, 3, 2)
      self.bn2 = nn.BatchNorm2d(zd//4)

      self.c3  = nn.Conv2d(zd//4, zd//2, 3, 1)
      self.bn3 = nn.BatchNorm2d(zd//2)

      self.c4  = nn.Conv2d(zd//2, zd, 3, 1)
      self.bn4 = nn.BatchNorm2d(zd)

      self.c5  = nn.Conv2d(zd, zd*2, 3, 1)
      self.bn5 = nn.BatchNorm2d(zd*2)
     
      self.c6  = nn.Conv2d(zd*2, zd*2, 3, 1)



    def forward(self, x):

        x = self.c1(x)
        x = self.bn1(x)
        x = self.lr(x)

        x = self.c2(x)
        x = self.bn2(x)
        x = self.lr(x)
        
        x = self.c3(x)
        x = self.bn3(x)
        x = self.lr(x)
        
        x = self.c4(x)
        x = self.bn4(x)
        x = self.lr(x)

        x = self.c5(x)

        return x


class GeneratorX(nn.Module):
    def __init__(self, zd=128, ch=1):
        super().__init__()


        self.conv1 = nn.ConvTranspose2d(zd, zd, 3, 1)
        self.bn1 = nn.BatchNorm2d(zd)
        self.rl  = nn.LeakyReLU(0.02)

        self.conv2 = nn.ConvTranspose2d(zd, zd//2, 3, 2)
        self.bn2   = nn.BatchNorm2d(zd//2)
        #    nn.LeakyReLU(0.02),

        self.conv3 =  nn.ConvTranspose2d(zd//2, zd//4, 3,2)
        self.bn3   = nn.BatchNorm2d(zd//4)
        #nn.LeakyReLU(0.02),
        
        self.conv4 =  nn.ConvTranspose2d(zd//4, zd//4, 3, 2)
        self.bn4   = nn.BatchNorm2d(zd//4)
        #nn.LeakyReLU(0.02),
        
        
        self.conv5 =  nn.ConvTranspose2d(zd//4, ch, 2, 1)
        self.tanh = nn.Tanh()
        
        

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.rl(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.rl(x)
        

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.rl(x)
        

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.rl(x)
        

        x = self.conv5(x)
        x = self.tanh(x)
        
        return x
        
