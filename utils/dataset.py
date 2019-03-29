import os

import torch
import numpy as np
from PIL import Image
from tempfile import mkdtemp
from torch.utils.data import Dataset, Subset
from torchvision.transforms import functional


class HoromaDataset(Dataset):

    def __init__(self, data_dir, split="train", subset=None, skip=0,
                 flattened=False, transforms=None):
        """
        Initialize the horoma dataset.

        :param data_dir: Path to the directory containing the samples.
        :param split: Which split to use. [train, valid, test]
        :param subset: Percentage size of dataset to use. Default: all.
        :param skip: How many element to skip before taking the subset.
        :param flattened: If True return the images in a flatten format.
        :param transforms: Transforms to apply on the dataset before using it.
        """
        self.nb_channels = 3
        self.height = 32
        self.width = 32
        self.datatype = "uint8"
        self.data_dir = data_dir
        self.str_labels = []
        self.str_to_id = []
        self.id_to_str = []

        self.train_labeled_split = ""

        if split == "train":
            self.nb_examples = 152000
        elif split == "train_labeled":
            self.nb_examples = 228
        elif split == "valid":
            self.nb_examples = 252
        elif split == "test":
            self.nb_examples = 498
        elif split == "train_overlapped":
            self.nb_examples = 548720
        elif split == "train_labeled_overlapped":
            self.nb_examples = 635
        elif split == "valid_overlapped":
            self.nb_examples = 696
        # else:
        #     raise ("Dataset: Invalid split. "
        #            "Must be [train, train_labeled, valid, test, train_overlapped, train_labeled_overlapped, valid_overlapped]")

        filename_x = os.path.join(data_dir, "{}_x.dat".format(split))

        if split.startswith("train"):
            if split == "train_all":
                split_1 = "train"
                self.train_labeled_split = "train_labeled"
                self.data, self.train_data, self.train_labeled = self.merge_memmap(
                    split_1, self.train_labeled_split, 152000, 228, split)
            elif split == "train_overlapped_all":
                split_1 = "train_overlapped"
                self.train_labeled_split = "train_labeled_overlapped"
                self.data, self.train_overlapped, self.train_labeled_overlapped = self.merge_memmap(
                    split_1, self.train_labeled_split, 548720, 635, split)
            else:
                self.data = np.memmap(
                    filename_x,
                    dtype=self.datatype,
                    mode="r",
                    shape=(self.nb_examples, self.height,
                           self.width, self.nb_channels)
                )

        # Get train data labels
        if split.startswith("train"):
            label_filename = os.path.join(
                data_dir, "{}_y.txt".format(self.train_labeled_split))
            self.targets = self.get_targets([label_filename])
        elif split == "valid_all":
            split_1 = "valid"
            split_2 = "valid_overlapped"
            self.data, self.valid, self.valid_overlapped = self.merge_memmap(
                split_1, split_2, 252, 696, split)
            filename_valid1 = os.path.join(
                data_dir, "{}_y.txt".format(split_1))
            filename_valid2 = os.path.join(
                data_dir, "{}_y.txt".format(split_2))
            # get valid labels
            self.targets = self.get_targets(
                [filename_valid1, filename_valid2])
        elif split == "valid" or split == "valid_overlapped":
            self.data = np.memmap(
                filename_x,
                dtype=self.datatype,
                mode="r",
                shape=(self.nb_examples, self.height,
                       self.width, self.nb_channels)
            )

            filename_y = os.path.join(
                data_dir, "{}_y.txt".format(split))

            # get valid labels
            self.targets = self.get_targets([filename_y])

        self.flattened = flattened

        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        img = self.data[index]
        label = None

        if self.transforms:
            img = self.transform(img)
        img = functional.to_tensor(img)

        if label is None:
            return img
        else:
            label = torch.Tensor([self.targets[index]])
            return img, label

    def get_targets(self, list_of_filename):
        targets = []

        for filename_y in list_of_filename:
            if os.path.exists(filename_y):
                pre_targets = np.loadtxt(filename_y, 'U2')

                self.str_labels = np.unique(pre_targets)

                self.str_to_id = dict(
                    zip(self.str_labels, range(len(self.str_labels))))
                self.id_to_str = dict((v, k)
                                      for k, v in self.str_to_id.items())

                targets += [self.str_to_id[_str]
                            for _str in pre_targets if _str in self.str_to_id]
        return np.asarray(targets)

    def merge_memmap(self, split_1, split_2, n_1, n_2, new_split):
        filename_1 = os.path.join(self.data_dir, "{}_x.dat".format(split_1))
        data1 = np.memmap(
            filename_1,
            dtype=self.datatype,
            mode="r",
            shape=(n_1, self.height, self.width, self.nb_channels)
        )
        filename_2 = os.path.join(self.data_dir, "{}_x.dat".format(split_2))
        data2 = np.memmap(
            filename_2,
            dtype=self.datatype,
            mode="r",
            shape=(n_2, self.height, self.width, self.nb_channels)
        )
        filename = os.path.join(self.data_dir, mkdtemp(), new_split + '.dat')
        data = np.memmap(
            filename,
            dtype=self.datatype,
            mode='w+',
            shape=(n_1 + n_2, self.height, self.width, self.nb_channels), order='C')
        data[:n_1, :] = data1
        data[n_1:, :] = data2
        return data, data1, data2


class CustomSubset(Dataset):
    """
    Not to be used, will fail miserably on a large dataset.
    """

    def __init__(self, dataset, indices):
        self.indices = indices

        self.dataset = dataset
        self.data = dataset.data[indices]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.dataset[self.indices[item]]

# TODO


class SplitDataset:

    def __init__(self, split=.9):
        """
        Callable class that performs a new split according to the region.

        Args:
            split (float): The proportion of examples to keep in the training set.
        """

        assert 0 < split < 1

        self.split = split

    def __call__(self, train_labeled_dataset, valid_dataset):
        """
        Takes train_labeled and valid dataset and returns a new split between training and validation.

        Args:
            train_labeled_dataset (torch.utils.data.Dataset): The original train_labeled dataset to split.
            valid_dataset (torch.utils.data.Dataset): The original valid dataset to split.
        Returns:
            train_set (torch.utils.data.Dataset): The new training set.
            valid_set (torch.utils.data.Dataset): The new validation set.
        """

        n_train = len(train_labeled_dataset)
        n_valid = len(valid_dataset)
        n_total = n_train + n_valid

        unique_regions, unique_region_inverse, unique_region_counts = np.unique(
            dataset.region_ids,
            return_counts=True,
            return_inverse=True
        )
        unique_regions = np.arange(unique_region_inverse.max() + 1)

        n_split = int(self.split * len(dataset))

        np.random.shuffle(unique_regions)
        cumsum = np.cumsum(unique_region_counts[unique_regions])

        last_region = np.argmax(1 * (cumsum > n_split))

        train_regions = unique_regions[:last_region]
        valid_regions = unique_regions[last_region:]

        indices = np.arange(n)

        train_indices = indices[np.isin(unique_region_inverse, train_regions)]
        valid_indices = indices[np.isin(unique_region_inverse, valid_regions)]

        train_set = Subset(dataset, train_indices)
        valid_set = Subset(dataset, valid_indices)

        return train_set, valid_set


class KFoldSplitDataset:

    def __init__(self, split=.9, permutation=1):
        """
        Callable class that performs a k-split according to the region.

        :param split: The proportion of examples to keep in the training set.
        :param permutation:
        """

        assert 0 < split < 1

        self.split = split
        self.permutation = permutation

    def __call__(self, dataset):
        """
        Takes a dataset and returns a split between training and validation.

        :param dataset: The original dataset to split.
        :return:
        train_set: The new training set.
        valid_set: The new validation set.
        """
        n = len(dataset)

        unique_regions, unique_region_inverse, unique_region_counts = np.unique(
            dataset.region_ids,
            return_counts=True,
            return_inverse=True
        )
        unique_regions = np.arange(unique_region_inverse.max() + 1)

        n_split = int(self.split * len(dataset))

        unique_regions = np.concatenate([
            unique_regions[self.permutation:],
            unique_regions[:self.permutation]
        ])

        cumsum = np.cumsum(unique_region_counts[unique_regions])

        last_region = np.argmax(1 * (cumsum > n_split))

        train_regions = unique_regions[:last_region]
        valid_regions = unique_regions[last_region:]

        indices = np.arange(n)

        train_indices = indices[np.isin(unique_region_inverse, train_regions)]
        valid_indices = indices[np.isin(unique_region_inverse, valid_regions)]

        train_set = Subset(dataset, train_indices)
        valid_set = Subset(dataset, valid_indices)

        return train_set, valid_set


class FullDataset(Dataset):

    def __init__(self, dataset):

        self.dataset = dataset
        self.dataset.transforms = None

        indices = np.arange(len(self)) % len(dataset)

        self.region_ids = self.dataset.region_ids[indices]
        self.targets = self.dataset.targets[indices]

    @staticmethod
    def transform(img, transform):

        img = functional.to_pil_image(img)

        if transform >= 11:
            transform += 1

        transforms = np.zeros((2 * 2 * 4))
        transforms[transform] = 1
        transforms.reshape((2, 2, 4))

        a = transform // (2 * 2)
        transform = transform % (2 * 2)
        h = transform // 2
        v = transform % 2

        if v == 1:
            img = functional.vflip(img)
        if h == 1:
            img = functional.hflip(img)

        angle = a * 90
        img = functional.rotate(img, angle)

        return img

    def __len__(self):
        return 15 * len(self.dataset)

    def __getitem__(self, item):

        transform = item // len(self.dataset)
        i = item % len(self.dataset)

        data = self.dataset[i]

        label = None

        if isinstance(data, tuple):
            img, label = data
        else:
            img = data

        img = self.transform(img, transform)
        img = functional.to_tensor(img)

        if label is None:
            return img
        else:
            return img, label


if __name__ == "__main__":
    train_dataset = HoromaDataset(
        data_dir='./../data/horoma',
        split='train_all',
        transforms=functional.to_pil_image
    )

    valid_dataset = HoromaDataset(
        data_dir='./../data/horoma',
        split='valid',
        transforms=functional.to_pil_image
    )

    # dataset = FullDataset(dataset)

    # loader = DataLoader(dataset, shuffle=False, batch_size=100)

    # splitter = SplitDataset(.9)

    # train, valid = splitter(dataset)

    print("No. of images in train dataset: ", len(train_dataset))
    print("No. of images in valid dataset: ", len(valid_dataset))
    # i = 152001
    # index = i - 152000
    # img = train_dataset[0][0]
    # label = train_dataset.targets[index]
    # print("label: ", train_dataset.id_to_str[int(label)])
    # img.show()
    # img.save("Sample", "JPEG")
