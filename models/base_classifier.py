import numpy as np
from sklearn.svm import SVC


class SVMClassifier:
	"""
	SVM classifier for multiclass classification
	"""
    def __init__(self):
        super(SVMClassifier, self).__init__()

        self.clf = SVC(kernel='rbf', gamma=.5)

    def train_classifier(self, train_X, train_y):
    	"""
    	train_X: low dim representation of images, size is (n_samples, dim)
    	train_y: target class of image, size is (n_samples)
    	"""
        self.clf.fit(train_X, train_y)
        y_pred = self.clf.predict(train_X)

        return y_pred
