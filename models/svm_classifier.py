import numpy as np
from sklearn.svm import SVC


class SVMClassifier:
    """
    SVM classifier for multiclass classification
    """

    def __init__(self, C=0.01, kernel='rbf', gamma=.5):
        super(SVMClassifier, self).__init__()

        self.clf = SVC(C=C, kernel=kernel, gamma=gamma)

    def train_classifier(self, train_X, train_y):
        """
        train_X: low dim representation of images, size is (n_samples, dim)
        train_y: target class of image, size is (n_samples)
        """
        train_X = train_X.cpu().numpy()
        self.clf.fit(train_X, train_y)
        y_pred = self.clf.predict(train_X)

        return y_pred

    def validate_classifier(self, valid_X):
        """
        valid_X: low dim representation of images, size is (n_samples, dim)
        valid_y: target class of image, size is (n_samples)
        """
        valid_X = valid_X.cpu().numpy()
        y_pred = self.clf.predict(valid_X)

        return y_pred
