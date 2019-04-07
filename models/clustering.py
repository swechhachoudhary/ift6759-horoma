from models.encoders import *
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
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

        self.ae1 = CVAE(latent_dim=10).apply(self.init_weights)
        self.ae2 = CVAE(latent_dim=10).apply(self.init_weights)
        self.ae3 = CVAE(latent_dim=10).apply(self.init_weights)
        self.ae4 = CVAE(latent_dim=10).apply(self.init_weights)
        self.ae5 = CVAE(latent_dim=10).apply(self.init_weights)
        self.ae6 = CVAE(latent_dim=10).apply(self.init_weights)
        self.ae7 = CVAE(latent_dim=10).apply(self.init_weights)
        self.ae8 = CVAE(latent_dim=10).apply(self.init_weights)
        self.ae9 = CVAE(latent_dim=10).apply(self.init_weights)
        self.ae10 = CVAE(latent_dim=10).apply(self.init_weights)
        self.ae11 = CVAE(latent_dim=10).apply(self.init_weights)
        self.ae12 = CVAE(latent_dim=10).apply(self.init_weights)
        self.ae13 = CVAE(latent_dim=10).apply(self.init_weights)
        self.ae14 = CVAE(latent_dim=10).apply(self.init_weights)
        self.ae15 = CVAE(latent_dim=10).apply(self.init_weights)
        self.ae16 = CVAE(latent_dim=10).apply(self.init_weights)
        self.ae17 = CVAE(latent_dim=10).apply(self.init_weights)

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
        ).apply(self.init_weights)

        self.output_layer_conv_net = nn.Linear(16 * 4 * 4, 17)
        torch.nn.init.kaiming_uniform_(self.output_layer_conv_net.weight)
        self.softmax_layer = nn.Softmax(dim=1)


    def init_weights(self, m):
        if type(m) == nn.Conv2d or type(m) == nn.Linear or type(m) == nn.ConvTranspose2d:
            torch.nn.init.kaiming_uniform_(m.weight)
            m.bias.data.fill_(0.01)


    def forward(self, inputs):
        output = self.clustering_network(inputs)
        output = output.view(-1, 16 * 4 * 4)
        output = self.output_layer_conv_net(output)
        return self.softmax_layer(output)


    def train_damic(self, inputs, batch_size):
        output_cluster_network = self(inputs)
        input_reconstruction_of_each_encoders = torch.FloatTensor(17, batch_size, 3, 32, 32).zero_()
        input_reconstruction_of_each_encoders[0],_,_ = self.ae1(inputs)
        input_reconstruction_of_each_encoders[1],_,_ = self.ae2(inputs)
        input_reconstruction_of_each_encoders[2],_,_ = self.ae3(inputs)
        input_reconstruction_of_each_encoders[3],_,_ = self.ae4(inputs)
        input_reconstruction_of_each_encoders[4],_,_ = self.ae5(inputs)
        input_reconstruction_of_each_encoders[5],_,_ = self.ae6(inputs)
        input_reconstruction_of_each_encoders[6],_,_ = self.ae7(inputs)
        input_reconstruction_of_each_encoders[7],_,_ = self.ae8(inputs)
        input_reconstruction_of_each_encoders[8],_,_ = self.ae9(inputs)
        input_reconstruction_of_each_encoders[9],_,_ = self.ae10(inputs)
        input_reconstruction_of_each_encoders[10],_,_ = self.ae11(inputs)
        input_reconstruction_of_each_encoders[11],_,_ = self.ae12(inputs)
        input_reconstruction_of_each_encoders[12],_,_ = self.ae13(inputs)
        input_reconstruction_of_each_encoders[13],_,_ = self.ae14(inputs)
        input_reconstruction_of_each_encoders[14],_,_ = self.ae15(inputs)
        input_reconstruction_of_each_encoders[15],_,_ = self.ae16(inputs)
        input_reconstruction_of_each_encoders[16],_,_ = self.ae17(inputs)

        return output_cluster_network, input_reconstruction_of_each_encoders
    

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
class RBFClustering:
    """clustering with Nonlinear SVM"""
    def __init__(self, seed):
        self.svm = SVC(C = 4e-4,random_state=seed)
        self.best_c = 0
    
    def train(self, data,labels):
        if type(data) is torch.Tensor:
            data = data.detach().cpu().numpy()
        self.svm.fit(data,labels)
        return self.svm

    def predict_cluster(self, data):
        if type(data) is torch.Tensor:
            data = data.detach().cpu().numpy()
        return self.svm.predict(data)
class SVMClustering:
    """clustering with LinearSVC"""
    def __init__(self, seed):
        self.svm = LinearSVC(C = 4e-4,random_state=seed)
        self.best_c = 0
    
    def train(self, data,labels):
        if type(data) is torch.Tensor:
            data = data.detach().cpu().numpy()
        self.svm.fit(data,labels)
        return self.svm

    def predict_cluster(self, data):
        if type(data) is torch.Tensor:
            data = data.detach().cpu().numpy()
        return self.svm.predict(data)

    def get_best_c(self,data,targets):

        for log_C in np.linspace(-20, 20, 50):
          if log_C < -10 or log_C > 0:
              continue
          C = np.exp(log_C)
          svm = LinearSVC(C=C)
          svm.fit(random_batch, random_targets.ravel())
          error_rate = 1 - np.mean([
              svm.score(validation_embeddings[1000 * i:1000 * (i + 1)],
                        np.array(validation_targets[1000 * i:1000 * (i + 1)]).ravel())
              for i in range(10)
          ])
          if error_rate < best_error_rate:
              best_error_rate = error_rate
              best_C = C
          print('C = {}, validation error rate = {} '.format(C, error_rate) +
                '(best is {}, {})'.format(best_C, best_error_rate))
        return best_C
