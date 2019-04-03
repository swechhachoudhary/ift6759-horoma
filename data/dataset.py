import os

import numpy as np
import torch
from PIL import Image
from tempfile import mkdtemp
from torchvision.transforms import functional
from torch.utils.data import Dataset

class LocalHoromaDataset(Dataset):
    """
    The data is not loaded from a file but given as parameters instead
    This dataset is use for the pretraining of the convolution clustering network
    Since this pretraining is supervised, we need a dataset who can handle targets
    """

    def __init__(self, data, targets):
        """
        Args:
            data : numpy array (number_of_sample, 3, 32, 32)
            targets : numpy array (number_of_sample, 1)
        """
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return torch.Tensor(self.data[index]) / 255, torch.Tensor([self.targets[index]])


class OriginalHoromaDataset(Dataset):

    def __init__(self, data_dir, split="train", subset=None, skip=0, flattened=False, transform=None):
        """
        Args:
            data_dir: Path to the directory containing the samples.
            split: Which split to use. [train, valid, test]
            subset: How many elements will be used. Default: all.
            skip: How many element to skip before taking the subset.
            flattened: If True return the images in a flatten format.
        """
        nb_channels = 3
        height = 32
        width = 32
        datatype = "uint8"

        if split == "train":
            self.nb_exemples = 152000
        elif split == "train_labeled":
            self.nb_exemples = 228
        elif split == "valid":
            self.nb_exemples = 252
        elif split == "test":
            self.nb_exemples = 498
        elif split == "train_overlapped":
            self.nb_exemples = 548720
        elif split == "train_labeled_overlapped":
            self.nb_exemples = 635
        elif split == "valid_overlapped":
            self.nb_exemples = 696
        else:
            raise (
                "Dataset: Invalid split. Must be [train, valid, test, train_overlapped, valid_overlapped]")

        filename_x = os.path.join(data_dir, "{}_x.dat".format(split))
        filename_y = os.path.join(data_dir, "{}_y.txt".format(split))

        self.targets = None
        if os.path.exists(filename_y):
            pre_targets = np.loadtxt(filename_y, 'U2')

            if subset is None:
                pre_targets = pre_targets[skip: None]
            else:
                pre_targets = pre_targets[skip: skip + subset]

            self.map_labels = np.unique(pre_targets)
            self.targets = np.asarray(
                [np.where(self.map_labels == t)[0][0] for t in pre_targets])

        self.data = np.memmap(filename_x, dtype=datatype, mode="r", shape=(
            self.nb_exemples, nb_channels, height, width))
        if subset is None:
            self.data = self.data[skip: None]
        else:
            self.data = self.data[skip: skip + subset]

        if flattened:
            self.data = self.data.reshape(len(self.data), -1)

        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.targets is not None:
            if self.transform:
                return torch.Tensor(self.transform(self.data[index])) / 255, torch.Tensor([self.targets[index]])
            else:
                return torch.Tensor(self.data[index]) / 255, torch.Tensor([self.targets[index]])
        if self.transform:
            return torch.Tensor(self.transform(self.data[index])) / 255
        else:
            return torch.Tensor(self.data[index]) / 255

class HoromaDataset(Dataset):

    def __init__(self, data_dir, split="train", subset=None, skip=0, flattened=False, transform=None):
        """
        Args:
            data_dir: Path to the directory containing the samples.
            split: Which split to use. [train, valid, test]
            subset: How many elements will be used. Default: all.
            skip: How many element to skip before taking the subset.
            flattened: If True return the images in a flatten format.
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
        elif split == "valid":
            self.nb_examples = 252
        elif split == "test":
            self.nb_examples = 498
        elif split == "train_overlapped":
            self.nb_exemples = 548720
        elif split == "valid_overlapped":
            self.nb_examples = 696
        # else:
        #     raise (
        #         "Dataset: Invalid split. Must be [train, valid, test, train_overlapped, valid_overlapped]")

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
        self.targets = None
        # Get train data labels
        if split.startswith("train"):
            label_filename = os.path.join(
                data_dir, "{}_y.txt".format(self.train_labeled_split))
            self.train_labels = self.get_targets([label_filename])
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

        if flattened:
            self.data = self.data.reshape(len(self.data), -1)

        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.targets is not None:
            if self.transform:
                return torch.Tensor(self.transform(self.data[index])) / 255, torch.Tensor([self.targets[index]])
            else:
                return torch.Tensor(self.data[index]) / 255, torch.Tensor([self.targets[index]])
        else:
            if self.transform:
                return torch.Tensor(self.transform(self.data[index])) / 255
            else:
                return torch.Tensor(self.data[index]) / 255

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

if __name__ == "__main__":
    train_dataset = HoromaDataset(
        data_dir='./../data/horoma',
        split='train_all'
    )

    valid_dataset = HoromaDataset(
        data_dir='./../data/horoma',
        split='valid'
    )
