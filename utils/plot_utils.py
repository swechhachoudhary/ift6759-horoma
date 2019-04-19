from PIL import Image
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics import confusion_matrix
from utils.dataset import HoromaDataset


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.savefig(title + '.png')
    plt.close()


def plot_historgrams(data, label, str_labels):
    """
    This function plots the histograms.
    """
    counter = Counter(data)
    print(counter)
    counter = dict(sorted(counter.items(), key=lambda i: i[0]))
    print(counter)
    frequencies = counter.values()
    names = counter.keys()
    print("{} frequencies: {},\n names: {}".format(label, frequencies, names))
    x_coordinates = np.arange(len(counter))
    plt.figure()
    plt.bar(x_coordinates, frequencies, align='center')
    plt.xticks(x_coordinates, str_labels)
    plt.title("Histogram of class labels for " + label + " labeled data")
    plt.xlabel("Class Ids")
    plt.ylabel("Frequency")
    plt.savefig(label + "_hist.png")
    plt.close()

if __name__ == "__main__":

    # plot bar graph of frequencies of classes
    plot_historgrams(valid.targets, "Validation", valid.str_labels)

    i = np.random.randint(0, len(train_labeled))
    print(i)
    print(train_labeled[i][0].size(), train_labeled[i][1])
    print(train_labeled[0][0].size(), train_labeled[0][1])

    img = Image.fromarray(
        (255 * train_labeled[i][0]).numpy().astype(np.uint8), 'RGB')
    img.show()
