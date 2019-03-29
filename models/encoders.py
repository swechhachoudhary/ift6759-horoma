from utils.model_utils import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA


class PCAEncoder:
    """
    Principal component analysis (PCA): Linear dimensionality
    reduction using Singular value decomposition of the data
    to project it to a lower dimensional space.

    """
    def __init__(self, seed):
        self.pca = PCA(n_components=0.9, random_state=seed)

    def fit(self, **kwargs):
        traindata = kwargs['data'].data
        return self.pca.fit_transform(traindata)

    def encode(self, encodingdata):
        if type(encodingdata) is torch.Tensor:
            encodingdata = encodingdata.cpu().numpy()
        return self.pca.transform(encodingdata)


class CVAE(nn.Module):
    """
    Convolutional autoencoder: composed of encoder an decoder components.
    Applies multiple 2D convolutions and 2D transpose convolutions, over
    the input images. each operator is followed by batch normalization and
    ReLU activations.

    :param input_dim: input dimension of the model, 3072.
    :param latent_dim: dimension of latent-space representation.

    """
    def __init__(self, input_dim=3072, latent_dim=2):
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.is_variational = True

        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),  # input is b, 3, 32, 32
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1),  # input is b, 3, 32, 32
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),  # b, 32, 8,8
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1),  # b, 32, 8,8
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),  # b, 16,4,4
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=4, stride=2, padding=1),  # b, 16,4,4
            nn.BatchNorm2d(16),
            nn.ReLU(),

        )
        self.embedding_mu = nn.Linear(16 * 4 * 4, self.latent_dim)
        self.embedding_sigma = nn.Linear(16 * 4 * 4, self.latent_dim)
        self.decode_embedding = nn.Linear(self.latent_dim, 16 * 4 * 4)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16, 32, kernel_size=4, stride=2, padding=1),  # b, 32, 8, 8
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, kernel_size=3, stride=1, padding=1),  # b, 32, 8, 8
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 64, kernel_size=4, stride=2, padding=1),  # b, 64, 16, 16
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1),  # b, 64, 16, 16
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),  # b, 3, 32, 32
            nn.BatchNorm2d(3),
            nn.Sigmoid()
        )

    def forward(self, x):
        """return reconstruction of the latent variable, the mean mu and log prob"""
        mu, logvar = self._get_dist_output(x)
        embedding = self._reparameterization_trick(mu, logvar)
        rev_embedding = F.relu(self.decode_embedding(embedding).view(-1, 16, 4, 4))
        return self.decoder(rev_embedding), mu, logvar

    def encode(self, input):
        """
        parametrizes the approximate posterior of the latent variables
        and outputs parameters to the distribution

        """
        return self._reparameterization_trick(*self._get_dist_output(input))

    def _get_dist_output(self, input):
        """return the two vectors of means and standard deviations"""
        input = self.encoder(input)
        input = input.view(-1, 16 * 4 * 4)
        mu, sigma = self.embedding_mu(input), self.embedding_sigma(input)

        return mu, sigma

    def decode(self, input):
        """reconstruct the input from the latent space representation"""
        rev_embedding = F.relu(self.decode_embedding(input).view(-1, 16, 4, 4))
        return self.decoder(rev_embedding)

    def _reparameterization_trick(self, mu, sigma):
        """
        Reparametrize samples such that the stochasticity is independent
        of the parameters. The reparametrization trick allow us backpropagate.

        """
        sigma = torch.exp(.5 * sigma)
        samples = torch.randn_like(sigma)
        return samples * sigma + mu

    def fit(self, data, batch_size, n_epochs, lr, device, experiment):
        """
        fit the model with the data.

        :param data: the input data
        :param batch_size: batch size set in config file
        :param seed: for reproductibililty.
        :param n_epochs: number of epochs
        :param lr: learning rate
        :param device: 'cuda' if available else 'cpu'
        :param experiment: for tracking comet experiment

        """
        train_loader, valid_loader = get_ae_dataloaders(data, batch_size, split=0.8)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        best_model = train_network(self, train_loader, valid_loader, optimizer, n_epochs, device, experiment)
        return encode_dataset(self, data, batch_size, device), best_model


class CAE(nn.Module):
    """
    Convolutional autoencoder: Applies a 2D convolutions, max pooling,
    and ReLU activations, two linear layers and 2D transpose convolutions
    over the input images.

    :param input_dim: input dimension of the model, 3072.
    :param latent_dim: dimension of latent-space representation.

    """
    def __init__(self, input_dim=3072, latent_dim=2):
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.is_variational = False

        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),  # input is b, 3, 32, 32
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1),  # input is b, 3, 32, 32
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),  # b, 32, 8,8
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1),  # b, 32, 8,8
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),  # b, 16,4,4
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=4, stride=2, padding=1),  # b, 16,4,4
            nn.BatchNorm2d(16),
            nn.ReLU(),

        )
        self.embedding = nn.Linear(16 * 4 * 4, self.latent_dim)
        self.decode_embedding = nn.Linear(self.latent_dim, 16 * 4 * 4)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16, 32, kernel_size=4, stride=2, padding=1),  # b, 32, 8, 8
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, kernel_size=3, stride=1, padding=1),  # b, 32, 8, 8
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 64, kernel_size=4, stride=2, padding=1),  # b, 64, 16, 16
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1),  # b, 64, 16, 16
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),  # b, 3, 32, 32
            nn.BatchNorm2d(3),
            nn.Sigmoid()
        )

    def forward(self, x):
        """return the reconstructed input after encoding/decoding"""
        embedding = self.encode(x)
        rev_embedding = F.relu(self.decode_embedding(embedding).view(-1, 16, 4, 4))
        return self.decoder(rev_embedding)

    def encode(self, x):
        return self.embedding(self.encoder(x).view(-1, 16 * 4 * 4))

    def fit(self, data, batch_size, n_epochs, lr, device, experiment):
        train_loader, valid_loader = get_ae_dataloaders(data, batch_size, split=0.8)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        best_model = train_network(self, train_loader, valid_loader, optimizer, n_epochs, device, experiment)
        return encode_dataset(self, data, batch_size, device), best_model


class VAE(nn.Module):
    """
    variational autoencoder: consists of an encoder an decoder components.
    continuous latent spaces allow easy random sampling
    and interpolation. Encoder compenent is outputting two vectors
    of size n: a vector of means, μ (mu), and another vector of standard
    deviations, σ (sigma).

    :param input_dim: input dimension of the model, 3072.
    :param latent_dim: dimension of latent-space representation.

    """

    def __init__(self, input_dim=3072, latent_dim=2):
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.is_variational = True
        self.d1 = (self.input_dim + self.latent_dim) // 6
        self.d2 = (self.d1 + self.latent_dim) // 2

        super().__init__()
        # Encoder network architecture
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.d1),
            nn.ReLU(),
            nn.Linear(self.d1, self.d2),
            nn.ReLU(),
        )
        self.latent_fc1 = nn.Linear(self.d2, self.latent_dim)
        self.latent_fc2 = nn.Linear(self.d2, self.latent_dim)

        # Decoder network architecture
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.d2),
            nn.ReLU(),
            nn.Linear(self.d2, self.d1),
            nn.ReLU(),
            nn.Linear(self.d1, self.input_dim),
            nn.Sigmoid()
        )

    def encode(self, input):
        """
        parametrizes the approximate posterior of the latent variables
        and outputs parameters to the distribution

        """
        return self._reparameterization_trick(*self._get_dist_output(input))

    def _get_dist_output(self, input):
        """get the two vectors of means and standard deviations"""
        input = input.view(-1, self.input_dim)
        input = self.encoder(input)
        mu, sigma = self.latent_fc1(input), self.latent_fc2(input)

        return mu, sigma

    def decode(self, input):
        """reconstruct the input from the latent space representation"""
        return self.decoder(input)

    def _reparameterization_trick(self, mu, sigma):
        """
        Reparametrize samples such that the stochasticity is independent
        of the parameters. The reparametrization trick allow us backpropagate.

        """
        sigma = torch.exp(.5 * sigma)
        samples = torch.randn_like(sigma)
        return samples * sigma + mu

    def forward(self, input):
        """return reconstruction of the latent variable, the mean mu and log prob"""
        mu, logvar = self._get_dist_output(input)
        z = self._reparameterization_trick(mu, logvar)
        reconstruction = self.decode(z)
        return reconstruction, mu, logvar

    def fit(self, data, batch_size, n_epochs, lr, device, experiment):
        train_loader, valid_loader = get_ae_dataloaders(data, batch_size, split=0.8)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        best_model = train_network(self, train_loader, valid_loader, optimizer, n_epochs, device, experiment)

        return encode_dataset(best_model, data, batch_size, device), best_model


class AE(nn.Module):
    """
    vanilla encoder: combination of linear layers which extent to the embedding layer,
    and then back to the output layer. the data is flattened,generating an input dimension
    of 3072, and then it is fed into the model, the embedding is used for clustering.

    :param input_dim: input dimension of the model, 3072.
    :param latent_dim: dimension of latent-space representation.

    """
    def __init__(self, input_dim=3072, latent_dim=2):
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.is_variational = False
        self.d1 = (self.input_dim + self.latent_dim) // 6
        self.d2 = (self.d1 + self.latent_dim) // 2

        super().__init__()
        # Encoder network architecture
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.d1),
            nn.ReLU(),
            nn.Linear(self.d1, self.d2),
            nn.ReLU(),
            nn.Linear(self.d2, self.latent_dim)
        )
        # Decoder network architecture
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.d2),
            nn.ReLU(),
            nn.Linear(self.d2, self.d1),
            nn.ReLU(),
            nn.Linear(self.d1, self.input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        """return the reconstructed input after encoding/decoding"""
        return self.decoder(self.encoder(x))

    def encode(self, x):
        """return encoded imput"""
        return self.encoder(x)

    def fit(self, data, batch_size, n_epochs, lr, device, experiment):
        train_loader, valid_loader = get_ae_dataloaders(data, batch_size, split=0.8)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        best_model = train_network(self, train_loader, valid_loader, optimizer, n_epochs, device, experiment)
        return encode_dataset(best_model, data, batch_size, device), best_model
