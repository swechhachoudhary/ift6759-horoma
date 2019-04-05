import os

import torch
import numpy as np
from torch.utils.data import Dataset, Subset
from torchvision.transforms import functional


class HoromaDataset(Dataset):

    def __init__(self, data_dir, split="train", subset=None, skip=0,
                 flattened=False, transforms=None):
        """
        Initialize the horoma dataset.

        :param data_dir: Path to the directory containing the samples.
        :param split: Which split to use. [train, valid, test]
        :param subset: Percentage size of dataset to use. Default: all.
        :param skip: How many element to skip before taking the subset.
        :param flattened: If True return the images in a flatten format.
        :param transforms: Transforms to apply on the dataset before using it.
        """
        nb_channels = 3
        height = 32
        width = 32
        datatype = "uint8"

        if split == "train":
            self.nb_examples = 150900
        elif split == "valid":
            self.nb_examples = 480
        elif split == "test":
            self.nb_examples = 498
        elif split == "train_overlapped":
            self.nb_examples = 544749
        elif split == "valid_overlapped":
            self.nb_examples = 1331
        else:
            raise ("Dataset: Invalid split. "
                   "Must be [train, valid, test, train_overlapped, valid_overlapped]")

        filename_x = os.path.join(data_dir, "{}_x.dat".format(split))
        filename_y = os.path.join(data_dir, "{}_y.txt".format(split))

        filename_region_ids = os.path.join(data_dir,
                                           "{}_regions_id.txt".format(split))
        self.region_ids = np.loadtxt(filename_region_ids, dtype=object)

        self.targets = None
        if os.path.exists(filename_y) and not split.startswith("train"):
            pre_targets = np.loadtxt(filename_y, 'U2')

            if subset is None:
                pre_targets = pre_targets[skip: None]
            else:
                pre_targets = pre_targets[skip: skip + subset]

            self.map_labels = np.unique(pre_targets)

            self.targets = np.asarray([
                np.where(self.map_labels == t)[0][0]
                for t in pre_targets
            ])

        self.data = np.memmap(
            filename_x,
            dtype=datatype,
            mode="r",
            shape=(self.nb_examples, height, width, nb_channels)
        )

        if subset is None:
            self.data = self.data[skip: None]
            self.region_ids = self.region_ids[skip: None]
        else:
            self.data = self.data[skip: skip + subset]
            self.region_ids = self.region_ids[skip: skip + subset]

        self.flattened = flattened

        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img = self.data[index]
        if self.transforms:
            img = self.transforms(img)

        if self.flattened:
            img = img.view(-1)

        if self.targets is not None:
            return img, torch.Tensor([self.targets[index]])
        return img


class CustomSubset(Dataset):
    """
    Not to be used, will fail miserably on a large dataset.
    """

    def __init__(self, dataset, indices):
        self.indices = indices

        self.dataset = dataset
        self.data = dataset.data[indices]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.dataset[self.indices[item]]


class SplitDataset:

    def __init__(self, split=.9):
        """
        Callable class that performs a split according to the region.

        Args:
            split (float): The proportion of examples to keep in the training set.
        """

        assert 0 < split < 1

        self.split = split

    def __call__(self, dataset):
        """
        Takes a dataset and returns a split between training and validation.

        Args:
            dataset (torch.utils.data.Dataset): The original dataset to split.

        Returns:
            train_set (torch.utils.data.Dataset): The new training set.
            valid_set (torch.utils.data.Dataset): The new validation set.
        """

        n = len(dataset)

        unique_regions, unique_region_inverse, unique_region_counts = np.unique(
            dataset.region_ids,
            return_counts=True,
            return_inverse=True
        )
        unique_regions = np.arange(unique_region_inverse.max() + 1)

        n_split = int(self.split * len(dataset))

        np.random.shuffle(unique_regions)
        cumsum = np.cumsum(unique_region_counts[unique_regions])

        last_region = np.argmax(1 * (cumsum > n_split))

        train_regions = unique_regions[:last_region]
        valid_regions = unique_regions[last_region:]

        indices = np.arange(n)

        train_indices = indices[np.isin(unique_region_inverse, train_regions)]
        valid_indices = indices[np.isin(unique_region_inverse, valid_regions)]

        train_set = Subset(dataset, train_indices)
        valid_set = Subset(dataset, valid_indices)

        return train_set, valid_set


class KFoldSplitDataset:

    def __init__(self, split=.9, permutation=1):
        """
        Callable class that performs a k-split according to the region.

        :param split: The proportion of examples to keep in the training set.
        :param permutation:
        """

        assert 0 < split < 1

        self.split = split
        self.permutation = permutation

    def __call__(self, dataset):
        """
        Takes a dataset and returns a split between training and validation.

        :param dataset: The original dataset to split.
        :return:
        train_set: The new training set.
        valid_set: The new validation set.
        """
        n = len(dataset)

        unique_regions, unique_region_inverse, unique_region_counts = np.unique(
            dataset.region_ids,
            return_counts=True,
            return_inverse=True
        )
        unique_regions = np.arange(unique_region_inverse.max() + 1)

        n_split = int(self.split * len(dataset))

        unique_regions = np.concatenate([
            unique_regions[self.permutation:],
            unique_regions[:self.permutation]
        ])

        cumsum = np.cumsum(unique_region_counts[unique_regions])

        last_region = np.argmax(1 * (cumsum > n_split))

        train_regions = unique_regions[:last_region]
        valid_regions = unique_regions[last_region:]

        indices = np.arange(n)

        train_indices = indices[np.isin(unique_region_inverse, train_regions)]
        valid_indices = indices[np.isin(unique_region_inverse, valid_regions)]

        train_set = Subset(dataset, train_indices)
        valid_set = Subset(dataset, valid_indices)

        return train_set, valid_set


class FullDataset(Dataset):

    def __init__(self, dataset):

        self.dataset = dataset
        self.dataset.transforms = None

        indices = np.arange(len(self)) % len(dataset)

        self.region_ids = self.dataset.region_ids[indices]
        self.targets = self.dataset.targets[indices]

    @staticmethod
    def transform(img, transform):

        img = functional.to_pil_image(img)

        if transform >= 11:
            transform += 1

        transforms = np.zeros((2 * 2 * 4))
        transforms[transform] = 1
        transforms.reshape((2, 2, 4))

        a = transform // (2 * 2)
        transform = transform % (2 * 2)
        h = transform // 2
        v = transform % 2

        if v == 1:
            img = functional.vflip(img)
        if h == 1:
            img = functional.hflip(img)

        angle = a * 90
        img = functional.rotate(img, angle)

        return img

    def __len__(self):
        return 15 * len(self.dataset)

    def __getitem__(self, item):

        transform = item // len(self.dataset)
        i = item % len(self.dataset)

        data = self.dataset[i]

        label = None

        if isinstance(data, tuple):
            img, label = data
        else:
            img = data

        img = self.transform(img, transform)
        img = functional.to_tensor(img)

        if label is None:
            return img
        else:
            return img, label




from torch.utils.data import DataLoader

class HoromaDataset(Dataset):

    def __init__(self, data_dir, split="train", subset=None, skip=0,
                 flattened=False, transforms=None):
        """
        Initialize the horoma dataset.

        :param data_dir: Path to the directory containing the samples.
        :param split: Which split to use. [train, valid, test]
        :param subset: Percentage size of dataset to use. Default: all.
        :param skip: How many element to skip before taking the subset.
        :param flattened: If True return the images in a flatten format.
        :param transforms: Transforms to apply on the dataset before using it.
        """
        nb_channels = 3
        height = 32
        width = 32
        datatype = "uint8"

        if split == "train":
            self.nb_examples = 150900
        elif split == "valid":
            self.nb_examples = 480
        elif split == "test":
            self.nb_examples = 498
        elif split == "train_overlapped":
            self.nb_examples = 544749
        elif split == "valid_overlapped":
            self.nb_examples = 1331
        else:
            raise ("Dataset: Invalid split. "
                   "Must be [train, valid, test, train_overlapped, valid_overlapped]")

        filename_x = os.path.join(data_dir, "{}_x.dat".format(split))
        filename_y = os.path.join(data_dir, "{}_y.txt".format(split))

        filename_region_ids = os.path.join(data_dir,
                                           "{}_regions_id.txt".format(split))
        self.region_ids = np.loadtxt(filename_region_ids, dtype=object)

        self.targets = None
        if os.path.exists(filename_y) and not split.startswith("train"):
            pre_targets = np.loadtxt(filename_y, 'U2')

            if subset is None:
                pre_targets = pre_targets[skip: None]
            else:
                pre_targets = pre_targets[skip: skip + subset]

            self.map_labels = np.unique(pre_targets)

            self.targets = np.asarray([
                np.where(self.map_labels == t)[0][0]
                for t in pre_targets
            ])

        self.data = np.memmap(
            filename_x,
            dtype=datatype,
            mode="r",
            shape=(self.nb_examples, height, width, nb_channels)
        )

        if subset is None:
            self.data = self.data[skip: None]
            self.region_ids = self.region_ids[skip: None]
        else:
            self.data = self.data[skip: skip + subset]
            self.region_ids = self.region_ids[skip: skip + subset]

        self.flattened = flattened

        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img = self.data[index]
        if self.transforms:
            img = self.transforms(img)

        if self.flattened:
            img = img.view(-1)

        if self.targets is not None:
            return img, torch.Tensor([self.targets[index]])
        return img

from torchvision import transforms

class RandomQuarterTurn:
    """
    Rotate the image by a multiple of a quarter turn.
    """

    def __call__(self, img):
        """
            img (PIL Image): Image to be rotated.
        Returns:
            PIL Image: Rotated image.
        """

        angle = np.random.choice([0, 90, 180, 270])

        return functional.rotate(img, angle)
class HoromaTransforms:
    """
    Performs all transforms at once.
    """

    def __init__(self):
        self.transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            RandomQuarterTurn(),
            transforms.ToTensor()
        ])

    def __call__(self, img):
        return self.transforms(img)


class FullDataset(Dataset):

    def __init__(self, dataset):

        self.dataset = dataset
        self.dataset.transforms = None

        indices = np.arange(len(self)) % len(dataset)

        self.region_ids = self.dataset.region_ids[indices]
        self.targets = self.dataset.targets[indices]

    @staticmethod
    def transform(img, transform):

        img = functional.to_pil_image(img)

        if transform >= 11:
            transform += 1

        transforms = np.zeros((2 * 2 * 4))
        transforms[transform] = 1
        transforms.reshape((2, 2, 4))

        a = transform // (2 * 2)
        transform = transform % (2 * 2)
        h = transform // 2
        v = transform % 2

        if v == 1:
            img = functional.vflip(img)
        if h == 1:
            img = functional.hflip(img)

        angle = a * 90
        img = functional.rotate(img, angle)

        return img

    def __len__(self):
        return 15 * len(self.dataset)

    def __getitem__(self, item):

        transform = item // len(self.dataset)
        i = item % len(self.dataset)

        data = self.dataset[i]

        label = None

        if isinstance(data, tuple):
            img, label = data
        else:
            img = data

        img = self.transform(img, transform)
        img = functional.to_tensor(img)

        if label is None:
            return img
        else:
            return img, label
          
 def calc_gradient_penalty(discriminator, real_data, fake_data,encoder1,encoder2, gp_lambda):
    """Calculate GP."""
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

def calc_gradient_penalty2(discriminator,real_data, fake_data, z1,z_enc1,z2, z_enc2,gp_lambda):
    """Calculate GP."""
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
def saveimages(Gxx1,Gxx2,Gzz1,Gzz2,noise1,noise2):
    save_image(Gxx2(noise2).data,
               os.path.join(IMAGE_PATH,'%d_2.png' % (epoch+1)),
               nrow=9, padding=1,
               normalize=False)
    e1 = Gxx2(reparameterize(Gxx1(noise1)))
    save_image(e1.data,
             os.path.join(IMAGE_PATH,'%d_1.png' % (epoch+1)),
             nrow=9, padding=1,
             normalize=False)
def test(Gxx1,Gxx2,Gzz1,Gzz2,epoch,from_z1):

        data=  next(iter(dataloader))
        data = data[0]
        data=data.to('cuda')

        if not from_z1:
            latent = Gzz1(data)  #z_hat

            z1 = reparameterize(latent)
            bbs = np.shape(data)[0]

            z_enc = Gzz2(z1)


            recon = Gxx2(reparameterize(Gxx1(reparameterize(z_enc))))



            n = min(data.size(0), 8)

            ss = np.shape(data)
            comparison = torch.cat([data[:n],
                                  recon.view(ss[0],ss[1],ss[2],ss[3])[:n]])
            save_image(comparison.cpu(),
                     IMAGE_PATH+'/reconstruction_z2_' + str(epoch) + '.png', nrow=n)
        else:

            latent = Gzz1(data)  #z_hat

            z1 = reparameterize(latent)


            recon = Gxx2(z1)



            n = min(data.size(0), 8)

            ss = np.shape(data)
            comparison = torch.cat([data[:n],
                                  recon.view(ss[0],ss[1],ss[2],ss[3])[:n]])
            save_image(comparison.cpu(),
                     IMAGE_PATH+'/reconstruction_z1_' + str(epoch) + '.png', nrow=n)
def reparameterize(encoded):
      zd = encoded.size(1)//2
      mu,logvar = encoded[:, :zd], encoded[:, zd:]
      std = torch.exp(0.5*logvar)
      eps = torch.randn_like(std)
      return eps.mul(std).add_(mu)

import math
import torch
from torch.optim import Optimizer


class OAdam(Optimizer):
    r"""Implements optimistic Adam algorithm.
    It has been proposed in `Training GANs with Optimism`_.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)
    .. _Training GANs with Optimism:
        https://arxiv.org/abs/1711.00141
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(OAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Adam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad.add_(group['weight_decay'], p.data)

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                # Optimistic update :)
                p.data.addcdiv_(step_size, exp_avg, exp_avg_sq.sqrt().add(group['eps']))

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                p.data.addcdiv_(-2.0 * step_size, exp_avg, denom)

        return loss


def prog_ali(e,b,b_total,loss_g,loss_d):
    sys.stdout.write("\r%3d: [%5d / %5d] G: %.4f D: %.4f" % (e,b,b_total,loss_g,loss_d))
    sys.stdout.flush()



def save_models():
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

import math
import torch
from torch.optim import Optimizer



class OptMirrorAdam(Optimizer):
    """Implements Optimistic Adam algorithm. Built on official implementation of Adam by pytorch. 
       See "Optimistic Mirror Descent in Saddle-Point Problems: Gointh the Extra (-Gradient) Mile"
       double blind review, paper: https://openreview.net/pdf?id=Bkg8jjC9KQ 

    Standard Adam 
    It has been proposed in `Adam: A Method for Stochastic Optimization`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_

    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(OptMirrorAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Adam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def step(self, closure=None):
       
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        
        loss = None
        
        # Do not allow training with out closure 
        if closure is not None:
            loss = closure()
        
        # Create a copy of the initial parameters 
        param_groups_copy = self.param_groups.copy()
        
        # ############### First update of gradients ############################################
        # ######################################################################################
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # @@@@@@@@@@@@@@@ State initialization @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg_1'] = torch.zeros_like(p.data)
                    state['exp_avg_2'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq_1'] = torch.zeros_like(p.data)
                    state['exp_avg_sq_2'] = torch.zeros_like(p.data)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq_1'] = torch.zeros_like(p.data)
                        state['max_exp_avg_sq_2'] = torch.zeros_like(p.data)
                # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ 
                        
                        
                        
                        
                exp_avg1, exp_avg_sq1 = state['exp_avg_1'], state['exp_avg_sq_1']
                if amsgrad:
                    max_exp_avg_sq1 = state['max_exp_avg_sq_1']
                beta1, beta2 = group['betas']

                
                # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ 
                # Step will be updated once  
                state['step'] += 1
                # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
                
                
                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg1.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq1.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                # *****************************************************
                # Additional steps, to get bias corrected running means  
                exp_avg1 = torch.div(exp_avg1, bias_correction1)
                exp_avg_sq1 = torch.div(exp_avg_sq1, bias_correction2)
                # *****************************************************
                                
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq1, exp_avg_sq1, out=max_exp_avg_sq1)
                    # Use the max. for normalizing running avg. of gradient
                    denom1 = max_exp_avg_sq1.sqrt().add_(group['eps'])
                else:
                    denom1 = exp_avg_sq1.sqrt().add_(group['eps'])

                step_size1 = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                p.data.addcdiv_(-step_size1, exp_avg1, denom1)


        
        # Perform additional backward step to calculate stochastic gradient - WATING STATE 
        loss = closure()
        
        #  
        # ############### Second evaluation of gradient step #######################################
        # ######################################################################################
        for (group, group_copy) in zip(self.param_groups,param_groups_copy ):
            for (p, p_copy) in zip(group['params'],group_copy['params']):
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                        
                        
                exp_avg2, exp_avg_sq2 = state['exp_avg_2'], state['exp_avg_sq_2']
                if amsgrad:
                    max_exp_avg_sq2 = state['max_exp_avg_sq_2']
                beta1, beta2 = group['betas']
                
                
                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg2.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq2.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                # *****************************************************
                # Additional steps, to get bias corrected running means  
                exp_avg2 = torch.div(exp_avg2, bias_correction1)
                exp_avg_sq2 = torch.div(exp_avg_sq2, bias_correction2)
                # *****************************************************
                                
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq2, exp_avg_sq2, out=max_exp_avg_sq2)
                    # Use the max. for normalizing running avg. of gradient
                    denom2 = max_exp_avg_sq2.sqrt().add_(group['eps'])
                else:
                    denom2 = exp_avg_sq2.sqrt().add_(group['eps'])

                step_size2 = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                p_copy.data.addcdiv_(-step_size2, exp_avg2, denom2)
                p = p_copy # pass parameters to the initial weight variables.
        return loss
        