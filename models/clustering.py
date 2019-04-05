from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.svm import LinearSVC
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
