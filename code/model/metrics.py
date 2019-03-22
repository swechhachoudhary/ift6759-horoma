import numpy as np


class Metric(object):
    """Base class for metric implementation"""
    def reset(self):
        """Resets all attributes of the class to 0"""
        for var in vars(self):
            setattr(self, var, 0)

    def dump(self):
        """Dump class attributes on a dictionary for JSON compatibility when saving

        Returns
        -------
        dict : dict
            Dictionary with attributes as keys and information as value.
        """
        return {var: str(getattr(self, var)) for var in vars(self)}


class LossAverage(Metric):
    """Running average over loss value"""
    def __init__(self):
        super()
        self.steps = None
        self.total = None

    def __call__(self, output, labels, params):
        pass

    def __str__(self):
        """Return average loss when printing the class

        Returns
        -------
        str : str
            String containing the average loss for the current epoch.
        """
        return "Loss: {:0.3f}\n".format(self.total / self.steps)

    def update(self, val):
        """Update the loss average with the latest value

        Parameters
        ----------
        val : float
            Latest value of the loss after evaluating the models results.
        """
        self.steps += 1
        self.total += val

    def get_average(self):
        return self.total / self.steps


class Accuracy(Metric):
    """Computes the accuracy for a given set of predictions"""
    def __init__(self):
        super()
        self.correct = None
        self.total = None

    def __call__(self, outputs, labels, params):
        """Updates the number of correct and total samples given the outputs of the model and the correct labels.

        Note:
        Adding the softmax function is not necessary to calculate the accuracy.

        Parameters
        ----------
        outputs : ndarray
            Model predictions.
        labels : ndarray
            Correct outputs.
        params : object
            Parameter class with general experiment information.
        """
        outputs = np.argmax(outputs, axis=1)
        self.total += float(labels.size)
        self.correct += np.sum(outputs == labels)

    def __str__(self):
        """Return sample accuracy when printing the class

        Returns
        -------
        str : str
            String containing the sample average accuracy for the summary steps.
        """
        return "Sample accuracy: {:0.3f} -- ({:.0f}/{:.0f})\n".format(self.correct/self.total, self.correct, self.total)

    def get_accuracy(self):
        """Returns average accuracy.

        Returns
        -------
        float : float
            Current accuracy for the sampled data.
        """
        return self.correct / self.total


class AccuracyPerClass(Metric):
    """Computes the accuracy per class"""

    def __init__(self):
        self.correct = None
        self.total = None

    def __call__(self, outputs, labels, params):
        """Updates the number of correct and total samples for each class given the outputs of the model and the correct
        labels.

        Parameters
        ----------
        outputs : ndarray
            Predictions of the model.
        labels : ndrarray
            Correct output.
        params : object
            Parameter class with general experiment information.
        """
        if not hasattr(self, 'num_classes'):
            self.num_classes = params.num_classes

        outputs = np.argmax(outputs, axis=1)
        for i in range(0, params.num_classes):
            indices = np.argwhere(labels == i)
            self.correct.setdefault(i, 0)
            self.correct[i] += sum(outputs[indices] == labels[indices])
            self.total.setdefault(i, 0)
            self.total[i] += len(labels[indices])

    def __str__(self):
        """Return per class accuracy when printing the class

        Returns
        -------
        str : str
            String containing the sample accuracy per class for the summary steps.
        """
        acc_str = ""
        for i in range(0, self.num_classes):
            correct = float(self.correct.get(i))
            total = float(self.total.get(i))
            class_accuracy = np.divide(correct, total, out=np.zeros_like(correct), where=total != 0).item()
            acc_str += "Class {} accuracy: {:0.3f} -- ({:.0f}/{:.0f})\n".format(i, class_accuracy, correct, total)
        return acc_str

    def reset(self):
        """Override reset method since we need dictionaries"""
        self.correct = {}
        self.total = {}

    def dump(self):
        """ Returns a string compatible with the dumps method of a JSON object. Override dump method since the attribute
        values will be dictionaries

        Returns
        -------
        class_accuracy : dict
            Dictionary containing the number of the class as key and the accuracy as value.
        """
        class_accuracy = {}
        for class_, correct in self.correct.items():
            class_accuracy[class_] = {
                'accuracy': str(correct / self.total[class_]),
                'correct': str(correct),
                'total': str(self.total[class_])
            }
        return class_accuracy


# Maintain all metrics required in this dictionary - these are used in the training and evaluation loops
# key accuracy must be kept to select best model in evaluation
metrics = {
    'train': {
        'accuracy': Accuracy(),
        'accuracy_per_class': AccuracyPerClass(),  # Includes sample accuracy
        'loss': LossAverage()
    },
    'eval': {
        'accuracy': Accuracy(),
        'accuracy_per_class': AccuracyPerClass(),  # Includes sample accuracy
    }
}
