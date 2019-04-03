from time import time
from data.dataset import HoromaDataset, OriginalHoromaDataset
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from utils.constants import Constants


def get_split_indices(targets, overlapped=False, split=0.7):
    """
    Compute splits by separating regions in labeled training and validation sets while also
    to the extent possible preserving class balance. Priority is given to separating regions.

    :param targets: Numpy array of labels.
    :param overlapped: Choice of validation set.
    :param split: Approximate percentage of data in labeled training set.
    :return: training indices, validation indices
    """
    if overlapped:
        regions = np.loadtxt(Constants.REGION_ID_PATH_OVERLAPPED)
    else:
        regions = np.loadtxt(Constants.REGION_ID_PATH)

    idx_map = _map_to_idx(regions)
    region_map = _map_to_regions(regions, targets)
    counts_map = _map_to_counts(regions, targets, idx_map)

    train_indices = []

    unique_targets, counts = np.unique(targets, return_counts=True)
    n_train_remain = counts // (1 / split)
    seen_regions = set([])

    for target in unique_targets:
        n_train = n_train_remain[target]
        while n_train > 0:
            region = _next_region_for_target(region_map, target, seen_regions)
            seen_regions.add(region)
            target_counts = counts_map[region]
            n_train_remain -= target_counts
            train_indices.extend(idx_map[region])
            n_train = n_train_remain[target]

    valid_indices = [i for i in range(len(targets)) if i not in train_indices]

    return train_indices, valid_indices


def _next_region_for_target(region_map, target, seen_regions):
    """
    Find an unexplored region which contains a sample for the given target.

    :param region_map: Pre-computed dictionary of target --> associated regions.
    :param target: Label in question.
    :param seen_regions: Set of previously explored regions. We want to avoid returning these twice.
    :return: Region id.
    """
    while True:
        region = region_map[target].pop()
        if region not in seen_regions:
            break

    return region


def _map_to_counts(regions, targets, idx_map):
    """
    Compute a mapping associating each region to an array of counts for each class.

    :param regions: array of region ids.
    :param targets: array of targets.
    :param idx_map: mapping for region --> indices
    :return: Dictionary associating each region to an array of counts for each class.
    """
    num_targets = len(np.unique(targets))
    mapping = {region: np.zeros(num_targets) for region in np.unique(regions)}
    for region in mapping:
        region_targets = targets[idx_map[region]]
        for t in region_targets:
            mapping[region][t] += 1
    return mapping


def _map_to_regions(regions, targets):
    """
    Helper function computing a mapping from labels to association regions.

    :param regions: Array of region ids.
    :param targets: Array of targets
    :return: dict for mapping target --> regions.
    """
    pairs = zip(targets, regions)
    mapping = {target: [] for target in targets}
    for key, val in pairs:
        if val not in mapping[key]:
            mapping[key].append(val)
    return mapping


def _map_to_idx(regions):
    """
    Helper function associating regions with their indices (samples)in the file.

    :param regions: Array of region ids.
    :return: Indices corresponding to this region.
    """
    idx = {elem: [] for elem in np.unique(regions)}
    for i, id in enumerate(regions):
        idx[id].append(i)
    return idx


def load_datasets(datapath, train_subset, flattened=False, overlapped=True):
    """
    Load Horoma datasets from specified data directory.

    :type datapath: str
    :type flattened: bool
    :type train_subset: str
    """

    print("Loading datasets from ({}) ...".format(datapath), end=' ')
    start_time = time()
    if overlapped:
        trainset = HoromaDataset(datapath, split="train_overlapped", subset=train_subset, flattened=flattened)
        labeled_set = HoromaDataset(datapath,
                                    split="valid_overlapped",
                                    flattened=flattened)
    else:
        trainset = OriginalHoromaDataset(datapath, split="train", subset=train_subset, flattened=flattened)
        labeled_set = OriginalHoromaDataset(datapath,
                                    split="valid",
                                    flattened=flattened)
    print("Done in {:.2f} sec".format(time() - start_time))

    return trainset, labeled_set

def load_original_horoma_datasets(datapath, train_subset, flattened=False, overlapped=True):
    """
    Load Original Horoma datasets from specified data directory.
    Return unlabeled, labeled and validation sets

    :type datapath: str
    :type flattened: bool
    :type train_subset: str
    :type overlapped: bool
    """

    print("Loading datasets from ({}) ...".format(datapath), end=' ')
    start_time = time()
    if overlapped:
        unlabeled_trainset = OriginalHoromaDataset(datapath, split="train_overlapped", subset=train_subset, flattened=flattened)
        labeled_trainset = OriginalHoromaDataset(datapath, split="train_labeled_overlapped", flattened=flattened)
        labeled_validset = OriginalHoromaDataset(datapath, split="valid_overlapped", flattened=flattened)

    else:
        unlabeled_trainset = OriginalHoromaDataset(datapath, split="train", subset=train_subset, flattened=flattened)
        labeled_trainset = OriginalHoromaDataset(datapath, split="train_labeled", flattened=flattened)
        labeled_validset = OriginalHoromaDataset(datapath, split="valid", flattened=flattened)

    print("Done in {:.2f} sec".format(time() - start_time))

    return unlabeled_trainset, labeled_trainset, labeled_validset


def assign_labels_to_clusters(model, data, labels_true):
    """
    Assign class label to each model cluster using labeled data.
    The class label is based on the class of majority samples within a cluster.
    Unassigned clusters are labeled as -1.
    """
    print("Assigning labels to clusters ...", end=' ')
    start_time = time()

    labels_pred = model.predict_cluster(data)
    labelled_clusters = []
    for i in range(model.n_clusters):
        idx = np.where(labels_pred == i)[0]
        if len(idx) != 0:
            labels_freq = np.bincount(labels_true[idx])
            labelled_clusters.append(np.argmax(labels_freq))
        else:
            labelled_clusters.append(-1)
    print("Done in {:.2f} sec".format(time() - start_time))

    return np.asarray(labelled_clusters)


def compute_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted")
    return accuracy, f1


def eval_model_predictions(model, x, y_true, cluster_labels):
    """
    Predict labels and compare to true labels to compute the accuracy.
    """
    print("Evaluating model ...", end=' ')
    start_time = time()
    y_pred = cluster_labels[model.predict_cluster(x)]

    accuracy, f1 = compute_metrics(y_true, y_pred)

    print(
        "Done in {:.2f} sec | Accuracy: {:.2f} - F1: {:.2f}".format(time() - start_time, accuracy * 100, f1 * 100))

    return y_pred, accuracy, f1


if __name__ == '__main__':
    train, labeled = load_datasets("/rap/jvb-000-aa/COURS2019/etudiants/data/horoma", None, overlapped=False)
