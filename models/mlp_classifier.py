import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class MLPClassifier(nn.Module):
    """
    MLP classifier for multiclass classification
    """

    def __init__(self, latent_dim=10, hidden_size=60, n_layers=2, n_class=17):
        """
        args:
        latent_dim: latent representation size
        hidden_size: number of hidden units in hidden layers
        n_layers: number of layers in MLP including output layer
        n_class: total number of classes
        """
        super(MLPClassifier, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        n_channels = [latent_dim] + [hidden_size] * (n_layers - 1) + [n_class]

        self.layers = nn.ModuleList(
            [nn.Linear(n_channels[i], n_channels[i + 1]) for i in range(n_layers)])
        self._weight_init()

    def _weight_init(self,):
        """
        Weight Ininitalization
        """
        k = math.sqrt(1. / self.hidden_size)
        for i in range(len(self.layers) - 1):
            nn.init.uniform_(self.layers[i].weight, -k, k)
            nn.init.constant_(self.layers[i].bias, 0.0)
        nn.init.uniform_(self.layers[-1].weight, -0.1, 0.1)
        nn.init.constant_(self.layers[-1].bias, 0.0)

    def forward(self, inputs):

        for i in range(self.n_layers - 1):
            out = F.relu(self.layers[i](inputs))
            inputs = out

        return self.layers[self.n_layers - 1](inputs)
