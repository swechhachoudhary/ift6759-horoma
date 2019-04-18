import torch
from torch.optim.optimizer import Optimizer, required

from torch.autograd import Variable
import torch.nn.functional as F
from torch import nn
from torch import Tensor
from torch.nn import Parameter

def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height,-1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False


    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)


    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)
      
class ResBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        nn.init.xavier_uniform_(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform_(self.conv2.weight.data, 1.)

        self.model = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(0.02),
#             nn.Upsample(scale_factor=2),
            self.conv1,
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.02),
            self.conv2,
            nn.BatchNorm2d(out_channels)
            )
        
        self.bypass = nn.Sequential()

    def forward(self, x):
      return self.model(x) + self.bypass(x)

class ResBlockTranspose(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlockTranspose, self).__init__()

        self.conv1 = nn.ConvTranspose2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = nn.ConvTranspose2d(out_channels, out_channels, 3, 1, padding=1)
        nn.init.xavier_uniform_(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform_(self.conv2.weight.data, 1.)

        self.model = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(0.02),
#             nn.Upsample(scale_factor=2),
            self.conv1,
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.02),
            self.conv2,
            nn.BatchNorm2d(out_channels)
            )
        
        self.bypass = nn.Sequential()

    def forward(self, x):
      return self.model(x) + self.bypass(x)

class GeneratorZ1(nn.Module):
    def __init__(self, zd=128, ch=1,zd1 = 64):
        super().__init__()

        self.lr  = nn.LeakyReLU(0.02)
        self.c1 = SpectralNorm(nn.Conv2d(ch, zd1//4, 3, 1,1))
        self.c2  = SpectralNorm(nn.Conv2d(zd1//4, zd1//2, 3,2,1))
        self.rn1 =  ResBlock(zd1//2, zd1//2)
        self.rn2 =  ResBlock(zd1//2,zd1//2)
        self.c3  = SpectralNorm(nn.Conv2d(zd1//2,zd1,3,2,1)) 
        self.bn3 = nn.BatchNorm2d(zd1)
        self.c4 = SpectralNorm(nn.Conv2d(zd1,zd1*2,3,1,1))
 
    def forward(self, x):
        x = self.lr(self.c1(x))
        x = self.c2(x)
        x = self.rn1(x)
        x = self.rn2(x)
        x = self.bn3(self.c3(x))
        x = self.c4(x)
        return x
class GeneratorZ2(nn.Module):
    def __init__(self, zd=128, zd1=64):
        super().__init__()
        self.c1 = SpectralNorm(nn.Conv2d(zd1, zd//2, 3, 2))
        self.bn1 = nn.BatchNorm2d(zd//2)
        self.lr  = nn.LeakyReLU(0.02)
        self.c2  = SpectralNorm(nn.Conv2d(zd//2, zd, 3, 1,1))
        self.rn1 =  ResBlock(zd, zd)
        self.c3  = SpectralNorm(nn.Conv2d(zd, zd*2, 3, 1))


    def forward(self, x):
        x = self.lr(self.bn1(self.c1(x)))
        x = self.c2(x)
        x = self.rn1(x)   
        x = self.c3(x)  
        
        return x

class GeneratorX1(nn.Module):
    def __init__(self, zd=128, ch=1,zd1=64):
        super().__init__()

        self.rl  = nn.LeakyReLU(0.02)

        
        self.conv1 = SpectralNorm(nn.ConvTranspose2d(zd, zd, 3, 1))

        self.rn1 =  ResBlockTranspose(zd, zd)
        
        self.conv2 = SpectralNorm(nn.ConvTranspose2d(zd, zd, 3, 1))
        self.bn2   = nn.BatchNorm2d(zd)

        self.conv3 =  SpectralNorm(nn.ConvTranspose2d(zd, zd1*2, 3,1))
        self.bn3   = nn.BatchNorm2d(zd1*2)
      
        self.conv4 =  SpectralNorm(nn.ConvTranspose2d(zd1*2, zd1*2, 2, 1))
        self.bn4   = nn.BatchNorm2d(zd1*2)
        
        self.conv5 =  SpectralNorm(nn.ConvTranspose2d(zd1*2, zd1*2, 2, 1))

    def forward(self, x):
        x = self.rl(self.conv1(x))
        x = self.rn1(x)
        x = self.rl(self.bn2(self.conv2(x)))
        x = self.rl(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        return x

class GeneratorX2(nn.Module):
    def __init__(self, zd=128, ch=1,zd1=64):
        super().__init__()

        self.rl  = nn.LeakyReLU(0.02)

        
        self.conv1 = SpectralNorm(nn.ConvTranspose2d(zd1, zd1, 3, 1,1))
        
        self.rn1 =  ResBlockTranspose(zd1, zd1)
        
      
        
        self.conv2 = SpectralNorm(nn.ConvTranspose2d(zd1, zd1, 3, 2,1))
        self.bn2   = nn.BatchNorm2d(zd1)
        
        
        
        self.conv3 =  SpectralNorm(nn.ConvTranspose2d(zd1, zd1//2, 3,1,1))
        self.bn3 = nn.BatchNorm2d(zd1//2)
        self.conv4 = SpectralNorm(nn.ConvTranspose2d(zd1//2,ch,2,1))
        
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.rl(self.conv1(x))
        x = self.rn1(x)
        x = self.rn1(x)
        x = nn.functional.interpolate(x,mode='bilinear',scale_factor=2,align_corners=False)
        x = self.rl(self.bn2(self.conv2(x)))
        x = self.rl(self.bn3(self.conv3(x)))
        x = self.tanh(self.conv4(x))
        return x


class GeneratorX1_interpolate(nn.Module):
    def __init__(self, zd=128, ch=1,zd1=64):
        super().__init__()

        self.rl  = nn.LeakyReLU(0.02)

        
        self.conv1 = SpectralNorm(nn.ConvTranspose2d(zd, zd, 3, 1,1))

        self.rn1 =  ResBlockTranspose(zd, zd)
        
        self.conv2 = SpectralNorm(nn.ConvTranspose2d(zd, zd, 3, 1,1))
        self.bn2   = nn.BatchNorm2d(zd)

        self.conv3 =  SpectralNorm(nn.ConvTranspose2d(zd, zd1*2, 3,1,1))
        self.bn3   = nn.BatchNorm2d(zd1*2)
      
        self.conv4 =  SpectralNorm(nn.ConvTranspose2d(zd1*2, zd1*2, 3, 1,1))

    def forward(self, x):
        x = self.rl(self.conv1(x))
        x = nn.functional.interpolate(x,mode='bilinear',scale_factor=2,align_corners=False)
        x = self.rn1(x)
        x = self.rl(self.bn2(self.conv2(x)))
        x = nn.functional.interpolate(x,mode='bilinear',scale_factor=2,align_corners=False)
        x = self.rl(self.bn3(self.conv3(x)))
        x = nn.functional.interpolate(x,mode='bilinear',scale_factor=2,align_corners=False)
        x = self.conv4(x)
        return x

class GeneratorX2_interpolate(nn.Module):
    def __init__(self, zd=128, ch=1,zd1=64):
        super().__init__()

        self.rl  = nn.LeakyReLU(0.02)
        self.conv1 = SpectralNorm(nn.ConvTranspose2d(zd1, zd1, 3, 1,1))
        self.rn1 =  ResBlockTranspose(zd1, zd1)
        self.conv2 = SpectralNorm(nn.ConvTranspose2d(zd1, zd1, 3, 1,1))
        self.bn2   = nn.BatchNorm2d(zd1)
        self.conv3 =  SpectralNorm(nn.ConvTranspose2d(zd1, zd1//2, 3,1,1))
        self.bn3 = nn.BatchNorm2d(zd1//2)
        self.conv4 = SpectralNorm(nn.ConvTranspose2d(zd1//2,ch,3,1,1))
        
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.rl(self.conv1(x))
        x = nn.functional.interpolate(x,mode='bilinear',scale_factor=2,align_corners=False)
        x = self.rn1(x)
        x = self.rn1(x)
        x = nn.functional.interpolate(x,mode='bilinear',scale_factor=2,align_corners=False)
        x = self.rl(self.bn2(self.conv2(x)))
        x = self.rl(self.bn3(self.conv3(x)))
        x = self.tanh(self.conv4(x))
        return x

class GeneratorX1_convolve(nn.Module):
    def __init__(self, zd=128, ch=1,zd1=64):
        super().__init__()

        self.rl  = nn.LeakyReLU(0.02)
        self.conv1 = SpectralNorm(nn.ConvTranspose2d(zd, zd, 4, 2,1))

        self.rn1 =  ResBlockTranspose(zd, zd)
        
        self.conv2 = SpectralNorm(nn.ConvTranspose2d(zd, zd, 4, 2,1))
        self.bn2   = nn.BatchNorm2d(zd)

        self.conv3 =  SpectralNorm(nn.ConvTranspose2d(zd, zd1*2, 4,2,1))
        self.bn3   = nn.BatchNorm2d(zd1*2)
      
        self.conv4 =  SpectralNorm(nn.ConvTranspose2d(zd1*2, zd1*2, 3, 1,1))

    def forward(self, x):
        x = self.rl(self.conv1(x))
        x = self.rn1(x)
        x = self.rl(self.bn2(self.conv2(x)))
        x = self.rl(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        return x

class GeneratorX2_convolve(nn.Module):
    def __init__(self, zd=128, ch=1,zd1=64):
        super().__init__()

        self.rl  = nn.LeakyReLU(0.02)
        self.conv1 = SpectralNorm(nn.ConvTranspose2d(zd1, zd1, 4, 2,1))        
        self.rn1 =  ResBlockTranspose(zd1, zd1)

        self.conv2 = SpectralNorm(nn.ConvTranspose2d(zd1, zd1, 4, 2,1))
        self.bn2   = nn.BatchNorm2d(zd1)

        self.conv3 =  SpectralNorm(nn.ConvTranspose2d(zd1, zd1//2, 3,1,1))
        self.bn3 = nn.BatchNorm2d(zd1//2)
        self.conv4 = SpectralNorm(nn.ConvTranspose2d(zd1//2,ch,3,1,1))
        
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.rl(self.conv1(x))
        x = self.rn1(x)
        x = self.rn1(x)
        x = self.rl(self.bn2(self.conv2(x)))
        x = self.rl(self.bn3(self.conv3(x)))
        x = self.tanh(self.conv4(x))
        return x



class Discriminator(nn.Module):
    def __init__(self,ch=3,ch2=128,zd=256,zd1 = 64):
        super(Discriminator, self).__init__()
        self.Dx  = DiscriminatorX(zd=zd,ch=ch,zd1=zd1)
        self.Dz1 = DiscriminatorZ1(zd=zd,ch=ch2,zd1 = zd1)
        self.Dxz = DiscriminatorXZ(zd=zd)
        
    def forward(self,x_input,z1_input,z2_input):   
        eps = 1e-5
        noise = Variable((torch.Tensor(x_input.size()).normal_(0, 0.1 * 0.01 ))).cuda()
        
        dx_out = self.Dx(x_input+noise)
      

        noise = Variable((torch.Tensor(dx_out.size()).normal_(0, 0.1 * 0.01))).cuda()
        
        Dz1_out = self.Dz1(torch.cat((dx_out,z1_input+noise),dim=1))
      

        d_out = self.Dxz(torch.cat((Dz1_out, z2_input), dim=1))+eps
       
        return d_out
      
class DiscriminatorX(nn.Module):
    def __init__(self, zd=128,ch=1,zd1 =64):
        super().__init__()
        
        
        self.lr  = nn.LeakyReLU(0.02)
        
        self.c1 = SpectralNorm(nn.Conv2d(ch, zd1//2, 3, 1))        
        self.c2  = SpectralNorm(nn.Conv2d(zd1//2, zd1, 3,2,1))
        self.c3 = SpectralNorm(nn.Conv2d(zd1,zd1,3,2,1))
        self.c4 = SpectralNorm(nn.Conv2d(zd1,zd1,3,1,1))
        self.c5 = SpectralNorm(nn.Conv2d(zd1,zd1,3,1,1))
       
        self.lr  = nn.LeakyReLU(0.02)
        self.d  = nn.Dropout2d(0.2)
            
    def forward(self, x):
        x = self.lr(self.c1(x))
        x = self.d(self.lr(self.c2(x)))
        x = self.d(self.lr(self.c3(x)))
        x = self.lr(self.c4(x))
        x = self.c5(x)        
        return x
      

class DiscriminatorZ1(nn.Module):
    def __init__(self, zd=128,ch=1,zd1=64):
        super().__init__()
        
        
        self.c1 = SpectralNorm(nn.Conv2d(zd1*2, zd1*2, 3, 2))
        self.lr  = nn.LeakyReLU(0.02)
        self.c2  = SpectralNorm(nn.Conv2d(zd1*2, zd, 3, 2))
        self.c3  = SpectralNorm(nn.Conv2d(zd, zd, 3, 1,1))
        self.c4  = SpectralNorm(nn.Conv2d(zd, zd, 3,1,1))
        self.c5  = SpectralNorm(nn.Conv2d(zd, zd,3,1,1))
        self.d  = nn.Dropout2d(0.5)
            
    def forward(self, x):
        x = self.lr(self.c1(x))  
        x = self.d(self.lr(self.c2(x)))
        x = self.lr(self.c3(x))
        x = self.d(self.lr(self.c4(x)))
        x = self.c5(x)
        return x


class DiscriminatorXZ(nn.Module):
    def __init__(self, zd=128):
        super().__init__()
        self.net = nn.Sequential(
            SpectralNorm(nn.Conv2d(zd*2, zd*4, 1, 1)),
            nn.LeakyReLU(0.02),
            nn.Dropout2d(0.2),
            SpectralNorm(nn.Conv2d(zd*4, zd, 1, 1)),    
            nn.LeakyReLU(0.02),
            nn.Dropout2d(0.2),
            SpectralNorm(nn.Conv2d(zd, 1, 1, 1)),
        )

    def forward(self, x):
        return self.net(x)
