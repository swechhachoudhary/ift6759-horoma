import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from collections import OrderedDict

cfg = {
    'AllConv13': [128, 128, 128, 'M', 256, 256, 256, 'M', 512, 256, 128, 'A'],
}

# Some utils class


class Reshape(nn.Module):
    """
    Flatten the output of the convolutional layer
    Parameters
    ----------
    Input shape: (N, C * W * H)
    Output shape: (N, C, W, H)
    """

    def __init__(self, shape, **kwargs):
        super(Reshape, self).__init__(**kwargs)
        self._shape = shape

    def forward(self, x):
        return x.reshape(x.size()[0], self._shape[0], self._shape[1], self._shape[2])


class BiasAdder(nn.Module):
    """
    Add a bias into the input
    """

    def __init__(self, channels, **kwargs):
        super(BiasAdder, self).__init__(**kwargs)
        self.bias = nn.Parameter(th.Tensor(1, channels, 1, 1))
        self.bias.data.uniform_(-0.1, 0.1)

    def forward(self, x):
        return x + self.bias


class Flatten(nn.Module):
    """
    Flatten 4D tensor into 2D tensor
    """

    def forward(self, x):
        return x.view(x.size(0), -1)


class Upaverage(nn.Module):
    """
    Upsample to reverse the avg pooling layer
    """

    def __init__(self, scale_factor, **kwargs):
        super(Upaverage, self).__init__(**kwargs)
        self.scale_factor = scale_factor
        self.upsample_layer = nn.Upsample(
            scale_factor=self.scale_factor, mode='nearest')

    def forward(self, x):
        return self.upsample_layer(x) * (1. / self.scale_factor)**2


def make_one_hot(labels, C=2):
    """
    Converts an integer label torch.autograd.Variable to a one-hot Variable.
    """
    target = th.eye(C)[labels.data]
    target = target.to(labels.get_device())
    return target

# Main NRM class


class NRM(nn.Module):

    def __init__(self, net_name, batch_size, num_class, use_bias=False, use_bn=False, do_topdown=False, do_pn=False, do_bnmm=False):
        super(NRM, self).__init__()
        self.num_class = num_class
        self.do_topdown = do_topdown
        self.do_pn = do_pn
        self.do_bnmm = do_bnmm
        self.use_bn = use_bn
        self.use_bias = use_bias
        self.batch_size = batch_size

        # create:
        # feature extractor in the forward cnn step: self.features
        # corresponding layer inm the top-down reconstruction nrm step: layers_nrm
        # instance norm used in the top-down reconstruction nrm step: insnorms_nrm
        # instance norm used in the forward cnn step: insnorms_cnn
        self.features, layers_nrm, insnorms_nrm, insnorms_cnn = self._make_layers(
            cfg[net_name], use_bias, use_bn, self.do_topdown)

        # create the classifer in the forward cnn step
        conv_layer = nn.Conv2d(in_channels=cfg[
                               net_name][-2], out_channels=self.num_class, kernel_size=(1, 1), bias=True)
        flatten_layer = Flatten()
        self.classifier = nn.Sequential(OrderedDict(
            [('conv', conv_layer), ('flatten', flatten_layer)]))

        # create the nrm
        if self.do_topdown:
            # add layers corresponding to the classifer in the forward step
            convtd_layer = nn.ConvTranspose2d(out_channels=cfg[
                                              net_name][-2], in_channels=self.num_class, kernel_size=(1, 1), stride=(1, 1), bias=False)
            convtd_layer.weight.data = conv_layer.weight.data
            layers_nrm += [('convtd', convtd_layer),
                           ('reshape', Reshape(shape=(self.num_class, 1, 1)))]

            self.nrm = nn.Sequential(OrderedDict(layers_nrm[::-1]))

            # if use path normalization, then also use instance normalization
            if self.do_pn:
                self.insnorms_nrm = nn.Sequential(
                    OrderedDict(insnorms_nrm[::-1]))
                self.insnorms_cnn = nn.Sequential(OrderedDict(insnorms_cnn))

    def forward(self, x, y=None):
        ahat = []
        that = []
        bcnn = []
        apn = []
        meancnn = []
        varcnn = []
        xbias = th.zeros([1, x.shape[1], x.shape[2], x.shape[3]],
                         device=x.get_device()) if self.do_pn else []
        insnormcnn_indx = 0

        # if do top-down reconstruction, we need to keep track of relu state, maxpool state,
        # mean and var of the activations, and the bias terms in the forward
        # cnn step
        if self.do_topdown:
            for name, layer in self.features.named_children():
                # keep track of the maxpool state
                if name.find('pool') != -1 and not name.find('average') != -1:
                    F.interpolate(layer(x), scale_factor=2, mode='nearest')
                    that.append(
                        th.gt(x - F.interpolate(layer(x), scale_factor=2, mode='nearest'), 0))
                    x = layer(x)
                    if self.do_pn:
                        xbias = layer(xbias)
                else:
                    x = layer(x)

                    if self.do_pn:  # get the forward results to compute the path normalization later
                        if name.find('batchnorm') != -1:
                            xbias = self.insnorms_cnn[insnormcnn_indx](xbias)
                            insnormcnn_indx += 1
                        else:
                            xbias = layer(xbias)
                    if name.find('relu') != -1:  # keep track of the relu state
                        ahat.append(th.gt(x, 0) + th.le(x, 0) * 0.1)
                        if self.do_pn:
                            apn.append(th.gt(xbias, 0) + th.le(xbias, 0) * 0.1)

                    if self.use_bn:
                        # keep track of the mean and var of the activations
                        if name.find('conv') != -1:
                            meancnn.append(
                                th.mean(x, dim=(0, 2, 3), keepdim=True))
                            varcnn.append(th.mean(
                                (x - th.mean(x, dim=(0, 2, 3), keepdim=True))**2, dim=(0, 2, 3), keepdim=True))
                        if self.use_bias:  # keep track of the bias terms when adding bias
                            if name.find('bias') != -1:
                                bcnn.append(layer.bias)
                        else:  # otherwise, keep track of the bias terms inside the batch norm
                            if name.find('batchnorm') != -1:
                                bcnn.append(layer.bias)
                    else:
                        if self.use_bias:
                            if name.find('conv') != -1:
                                bcnn.append(layer.bias)

            # reverse the order of the parameters/variables that we keep track
            # to use in the top-down reconstruction nrm step since nrm is the
            # reverse of cnn
            ahat = ahat[::-1]
            that = that[::-1]
            bcnn = bcnn[::-1]
            apn = apn[::-1]
            meancnn = meancnn[::-1]
            varcnn = varcnn[::-1]
        else:
            x = self.features(x)

        # send the features into the classifier
        z = self.classifier(x)

        # do reconstruction via nrm
        # xhat: the reconstruction image
        # loss_pn: path normalization loss
        # loss_bnmm: batch norm moment matching loss
        if self.do_topdown:
            xhat, _, loss_pn, loss_bnmm = self.topdown(self.nrm, make_one_hot(y, self.num_class), ahat, that, bcnn, th.ones([1, z.size()[1]], device=z.get_device()), apn, meancnn, varcnn) if y is not None else self.topdown(
                self.nrm, make_one_hot(th.argmax(z.detach(), dim=1), self.num_class), ahat, that, bcnn, th.ones([1, z.size()[1]], device=z.get_device()), apn, meancnn, varcnn)
        else:
            xhat = None
            loss_pn = None
            loss_bnmm = None

        return [z, xhat, loss_pn, loss_bnmm]

    def _make_layers(self, cfg, use_bias, use_bn, do_topdown):
        layers = []
        layers_nrm = []
        insnorms_nrm = []
        insnorms_cnn = []
        in_channels = 3

        for i, x in enumerate(cfg):
            # if max pooling layer, then add max pooling and dropout into the
            # cnn. Add upsample layers, dropout, batchnorm, and instance norm -
            # for path normaliztion - into the nrm.
            if x == 'M':
                layers += [('pool%i' % i, nn.MaxPool2d(2, stride=2)),
                           ('dropout%i' % i, nn.Dropout(0.5))]
                if do_topdown:
                    if use_bn:
                        layers_nrm += [('upsample%i' % i, nn.Upsample(scale_factor=2, mode='nearest')),
                                       ('dropout%i' % i, nn.Dropout(0.5)), ('batchnorm%i' % i, nn.BatchNorm2d(cfg[i - 1]))]
                        insnorms_nrm += [('instancenormtd%i' %
                                          i, nn.InstanceNorm2d(cfg[i - 1], affine=True))]
                    else:
                        layers_nrm += [('upsample%i' % i, nn.Upsample(scale_factor=2,
                                                                      mode='nearest')), ('dropout%i' % i, nn.Dropout(0.5))]

            # if avg pooling layer, then add average pooling layer into the
            # cnn. Add up average layers, batchnorm and instance norm - for
            # path normaliztion - into the nrm.
            elif x == 'A':
                layers += [('average%i' % i, nn.AvgPool2d(6, stride=1))]
                if do_topdown:
                    if use_bn:
                        layers_nrm += [('upaverage%i' % i, Upaverage(scale_factor=6)),
                                       ('batchnorm%i' % i, nn.BatchNorm2d(cfg[i - 1]))]
                        insnorms_nrm += [('instancenormtd%i' %
                                          i, nn.InstanceNorm2d(cfg[i - 1], affine=True))]
                    else:
                        layers_nrm += [('upaverage%i' %
                                        i, Upaverage(scale_factor=6))]

            else:  # add other layers into the cnn and the nrm
                padding_cnn = (0, 0) if x == 512 else (1, 1)
                padding_nrm = (0, 0) if x == 512 else (1, 1)
                if use_bn:
                    conv_layer = nn.Conv2d(in_channels=in_channels, out_channels=x, kernel_size=(
                        3, 3), padding=padding_cnn, bias=False)
                    if use_bias:
                        layers += [('conv%i' % i, conv_layer),
                                   ('batchnorm%i' % i, nn.BatchNorm2d(x)),
                                   ('bias%i' % i, BiasAdder(channels=x)),
                                   ('relu%i' % i, nn.LeakyReLU(0.1))]
                    else:
                        layers += [('conv%i' % i, conv_layer),
                                   ('batchnorm%i' % i, nn.BatchNorm2d(x)),
                                   ('relu%i' % i, nn.LeakyReLU(0.1))]

                    insnorms_cnn += [('instancenormcnn%i' %
                                      i, nn.InstanceNorm2d(x, affine=True))]
                    if do_topdown:
                        if (cfg[i - 1] == 'M' or cfg[i - 1] == 'A') and not i == 0:
                            layers_nrm += [('convtd%i' % i, nn.ConvTranspose2d(out_channels=in_channels, in_channels=x, kernel_size=3, stride=(1, 1),
                                                                               padding=padding_nrm, bias=False))]
                            layers_nrm[-1][-1].weight.data = conv_layer.weight.data
                        else:
                            layers_nrm += [('batchnormtd%i' % i, nn.BatchNorm2d(in_channels)), ('convtd%i' % i, nn.ConvTranspose2d(
                                out_channels=in_channels, in_channels=x, kernel_size=3, stride=(1, 1), padding=padding_nrm, bias=False))]
                            layers_nrm[-1][-1].weight.data = conv_layer.weight.data
                            insnorms_nrm += [('instancenormtd%i' %
                                              i, nn.InstanceNorm2d(in_channels, affine=True))]

                elif use_bias:
                    conv_layer = nn.Conv2d(in_channels=in_channels, out_channels=x, kernel_size=(
                        3, 3), padding=padding_cnn, use_bias=True)
                    layers += [('conv%i' % i, conv_layer),
                               ('relu%i' % i, nn.LeakyReLU(0.1))]
                    if do_topdown:
                        layers_nrm += [('convtd%i' % i, nn.ConvTranspose2d(out_channels=in_channels, in_channels=x, kernel_size=3, stride=(1, 1),
                                                                           padding=padding_nrm, bias=False))]
                        layers_nrm[-1][-1].weight.data = conv_layer.weight.data

                else:
                    conv_layer = nn.Conv2d(in_channels=in_channels, out_channels=x, kernel_size=(
                        3, 3), padding=padding_cnn, bias=False)
                    layers += [('conv%i' % i, conv_layer),
                               ('relu%i' % i, nn.LeakyReLU(0.1))]
                    if do_topdown:
                        layers_nrm += [('convtd%i' % i, nn.ConvTranspose2d(out_channels=in_channels, in_channels=x, kernel_size=3, stride=(1, 1),
                                                                           padding=padding_nrm, bias=False))]
                        layers_nrm[-1][-1].weight.data = conv_layer.weight.data

                in_channels = x

        model = nn.Sequential(OrderedDict(layers))

        return model, layers_nrm, insnorms_nrm, insnorms_cnn

    def topdown(self, net, xhat, ahat, that, bcnn, xpn, apn, meancnn, varcnn):
        mu = xhat
        mupn = xpn
        loss_pn = th.zeros([self.batch_size, ], device=mu.get_device())
        loss_bnmm = th.zeros([self.batch_size, ], device=mu.get_device())

        ahat_indx = 0
        that_indx = 0
        meanvar_indx = 0
        insnormtd_indx = 0
        prev_name = ''

        for i, (name, layer) in enumerate(net.named_children()):
            if name.find('conv') != -1 and i > 1:
                # mask the intermediate rendered images by the relu states in
                # the forward step
                mu = mu * \
                    ahat[ahat_indx].type(th.FloatTensor).to(mu.get_device())

                if self.do_pn:  # compute the path normalization loss
                    mupn = mupn * \
                        apn[ahat_indx].type(th.FloatTensor).to(mu.get_device())
                    mu_b = bcnn[ahat_indx].data.reshape((1, -1, 1, 1)) * mu
                    mupn_b = bcnn[ahat_indx].data.reshape((1, -1, 1, 1)) * mupn

                    loss_pn_layer = th.mean(
                        th.abs(mu_b - mupn_b), dim=(1, 2, 3))
                    loss_pn = loss_pn + loss_pn_layer

                ahat_indx += 1

            if prev_name.find('upsamplelayer') != -1 and not prev_name.find('avg') != -1:
                # mask the intermediate rendered images by the maxpool states
                # in the forward step
                mu = mu * \
                    that[that_indx].type(th.FloatTensor).to(mu.get_device())
                if self.do_pn:
                    mupn = mupn * \
                        that[that_indx].type(
                            th.FloatTensor).to(mu.get_device())
                that_indx += 1

            # compute the next intermediate rendered images
            mu = layer(mu)

            # compute the next intermediate rendered results for computing the
            # path normalization loss in the next layer
            if (name.find('batchnorm') != -1) and (i < len(net) - 1):
                if self.do_pn:
                    mupn = self.insnorms_nrm[insnormtd_indx](mupn)
                    insnormtd_indx += 1
            else:
                if self.do_pn:
                    mupn = layer(mupn)

            if (name.find('conv') != -1) and (i != (len(net) - 2)):
                if self.do_bnmm and self.use_bn:
                    # compute the KL distance between two Gaussians - the
                    # intermediate rendered images and the mean/var from the
                    # forward step
                    loss_bnmm = loss_bnmm + 0.5 * th.mean(((th.mean(mu, dim=(0, 2, 3)) - meancnn[meanvar_indx])**2) / varcnn[meanvar_indx]) + 0.5 * th.mean(th.mean((mu - th.mean(mu, dim=(0, 2, 3), keepdim=True))**2, dim=(
                        0, 2, 3)) / varcnn[meanvar_indx]) - 0.5 * th.mean(th.log(th.mean((mu - th.mean(mu, dim=(0, 2, 3), keepdim=True))**2, dim=(0, 2, 3)) + 1e-8) - th.log(varcnn[meanvar_indx])) - 0.5
                    meanvar_indx += 1

            prev_name = name

        return mu, mupn, loss_pn, loss_bnmm
