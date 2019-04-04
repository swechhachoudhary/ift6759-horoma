from models.encoders import *
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class DAMICClustering(nn.Module):
    """
        Clustering network for DAMIC
        Each cluster is reprensented by an autoencoder
        A convolutional network assign one input to a specific autoencoder
        See 'Deep clustering based on a mixture of autoencoders' by Chazan, Gannot and Goldberger
    """
    def __init__(self, n_clusters):
        super().__init__()
        self.n_clusters = n_clusters
        self.autoencoders = np.array([])
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        for i in range(n_clusters):
            encoding_model = CVAE(latent_dim=10).to(device)
            self.autoencoders = np.append(self.autoencoders, np.array([encoding_model]))
        self.clustering_network = nn.Sequential(
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
            nn.ReLU()
        )
        self.output_layer_conv_net = nn.Linear(16 * 4 * 4, 17)

    def train_clustering_network(self, input):
        self.clustering_network.train()
        self.output_layer_conv_net.train()
        output = self.clustering_network(input)
        output = output.view(-1, 16 * 4 * 4)
        output = self.output_layer_conv_net(output)
        return F.softmax(output, dim=1)
    
    def test_clustering_network(self, inputs):
        self.clustering_network.eval()
        self.output_layer_conv_net.eval()
        output = self.clustering_network(inputs)
        output = output.view(-1, 16 * 4 * 4)
        output = self.output_layer_conv_net(output)
        return F.softmax(output, dim=1)        

    def train_damic(self, input, batch_size):
        output_cluster_network = self.train_clustering_network(input)
        input_reconstruction_of_each_encoders = torch.FloatTensor(17, batch_size, 3, 32, 32).zero_()
        for i in range(self.n_clusters):
            encoded_decoded_x, _, _ = self.autoencoders[i](input)
            input_reconstruction_of_each_encoders[i] = encoded_decoded_x
        return output_cluster_network, input_reconstruction_of_each_encoders
        
    def forward(self, x):
        print("Warning : forward should not be called for DAMICClustering")
        print("Instead call train_clustering_network or train_damic")
    
class KMeansClustering:
    """clustering with K-means"""
    def __init__(self, n_clusters, seed):
        self.kmeans = KMeans(init="k-means++", n_clusters=n_clusters, n_init=5, max_iter=1000, random_state=seed,
                             n_jobs=-1)
        self.n_clusters = n_clusters

    def train(self, data):
        if type(data) is torch.Tensor:
            data = data.detach().cpu().numpy()
        self.kmeans.fit(data)
        return self.kmeans

    def predict_cluster(self, data):
        if type(data) is torch.Tensor:
            data = data.detach().cpu().numpy()
        return self.kmeans.predict(data)


class GMMClustering:
    """clustering with gaussian mixture model"""
    def __init__(self, n_clusters, seed):
        self.gmm = GaussianMixture(n_components=n_clusters, random_state=seed)
        self.n_clusters = n_clusters

    def train(self, data):
        if type(data) is torch.Tensor:
            data = data.detach().cpu().numpy()
        self.gmm.fit(data)
        return self.gmm

    def predict_cluster(self, data):
        if type(data) is torch.Tensor:
            data = data.detach().cpu().numpy()
        return self.gmm.predict(data)
