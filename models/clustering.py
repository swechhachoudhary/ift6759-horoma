from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import torch


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
