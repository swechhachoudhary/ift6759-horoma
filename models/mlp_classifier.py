import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPClassifier(nn.Module):
    """
    MLP classifier for multiclass classification
    """

    def __init__(self, latent_dim=10, hidden_size=60, n_layers=2, n_class=17):
        super(MLPClassifier, self).__init__()
        self.n_layers = n_layers
        n_channels = [latent_dim] + [hidden_size] * (n_layers - 1) + [n_class]

        self.layers = nn.ModuleList(
            [nn.Linear(n_channels[i], n_channels[i + 1]) for i in range(n_layers)])

    def _weight_init(self,):
        for layer in self.layers:
            nn.init.constant_(layer.weight, 0.0)
            nn.init.constant_(layer.bias, 0.0)

    def forward(self, inputs):

        for i in range(self.n_layers - 1):
            out = F.relu(self.layers[i](inputs))
            inputs = out

        return self.layers[self.n_layers - 1](inputs)
