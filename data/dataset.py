import os

import numpy as np
import torch
from PIL import Image
from tempfile import mkdtemp
from torchvision.transforms import functional
from torch.utils import data
from torch.utils.data import Dataset
import matplotlib.pyplot as plt


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
        elif split == "valid":
            self.nb_exemples = 252
        elif split == "test":
            self.nb_exemples = 498
        elif split == "train_overlapped":
            self.nb_exemples = 548720
        elif split == "valid_overlapped":
            self.nb_exemples = 696
        else:
            raise (
                "Dataset: Invalid split. Must be [train, valid, test, train_overlapped, valid_overlapped]")

        filename_x = os.path.join(data_dir, "{}_x.dat".format(split))
        filename_y = os.path.join(data_dir, "{}_y.txt".format(split))

        self.targets = None
        if os.path.exists(filename_y) and not split.startswith("train"):
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

        self.splits = ["train", "train_labeled", "valid", "train_overlapped",
                       "train_labeled_overlapped", "valid_overlapped"]

        self.splits_with_all = {"train_all": ["train", "train_labeled"],
                                "train_overlapped_all": ["train_overlapped", "train_labeled_overlapped"],
                                "valid_all": ["valid", "valid_overlapped"],
                                "train_unlabeled_all": ["train", "train_overlapped"],
                                "train_labeled_all": ["train_labeled", "train_labeled_overlapped"]}

        self.nb_examples = {"train": 152000,
                            "train_labeled": 228,
                            "valid": 252,
                            "train_overlapped": 548720,
                            "train_labeled_overlapped": 635,
                            "valid_overlapped": 696,
                            "train_all": 152228,
                            "train_overlapped_all": 549355,
                            "valid_all": 948,
                            "train_unlabeled_all": 700720,
                            "train_labeled_all": 863}

        filename_x = os.path.join(data_dir, "{}_x.dat".format(split))
        if split in self.splits:
            self.data = np.memmap(
                filename_x,
                dtype=self.datatype,
                mode="r",
                shape=(self.nb_examples[split], self.height,
                       self.width, self.nb_channels)
            )
        elif split in self.splits_with_all:
            self.data = self.merge_memmap(self.splits_with_all[split][
                                          0], self.splits_with_all[split][1], split)
        elif split == "test":
            self.nb_examples = 498

        self.data = self.data.reshape(
            len(self.data), self.nb_channels, self.height, self.width)

        self.targets = None

        if split in ["train_labeled", "train_labeled_overlapped", "valid", "valid_overlapped"]:
            filename_y = os.path.join(
                data_dir, "{}_y.txt".format(split))
            self.targets = self.get_targets([filename_y])
        elif split in ["valid_all", "train_labeled_all"]:
            filename_y = [os.path.join(data_dir, "{}_y.txt".format(
                split_i)) for split_i in self.splits_with_all[split]]
            self.targets = self.get_targets(filename_y)

        if flattened:
            self.data = self.data.reshape(len(self.data), -1)

        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.transform:
            img = torch.Tensor(self.transform(self.data[index])) / 255
        else:
            img = torch.Tensor(self.data[index]) / 255
        # img = img.transpose(2, 0).transpose(2, 1)
        if self.targets is None:
            return img
        else:
            label = torch.Tensor([self.targets[index]])
            return img, label

    def merge_memmap(self, split_1, split_2, new_split):
        n_1 = self.nb_examples[split_1]
        n_2 = self.nb_examples[split_2]
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
        data[:n_1] = data1
        data[n_1:] = data2
        return data

    def get_targets(self, list_of_filename):
        targets = []
        if os.path.exists(list_of_filename[0]):
            pre_targets = np.loadtxt(list_of_filename[0], 'U2')

            self.str_labels = np.unique(pre_targets)

            self.str_to_id = dict(
                zip(self.str_labels, range(len(self.str_labels))))
            self.id_to_str = dict((v, k)
                                  for k, v in self.str_to_id.items())

        for filename_y in list_of_filename:
            pre_targets = np.loadtxt(filename_y, 'U2')
            targets += [self.str_to_id[_str]
                        for _str in pre_targets if _str in self.str_to_id]
        return np.asarray(targets)

if __name__ == "__main__":
    valid = HoromaDataset(
        data_dir='./../data/horoma',
        split='valid'
    )

    train_labeled = HoromaDataset(
        data_dir='./../data/horoma',
        split='train_labeled'
    )

    print(len(valid))
    print(len(valid.targets))
    print(len(train_labeled))

    # i = np.random.randint(0, len(train_labeled))
    # print(i)
    # print(train_labeled[i][0].size(), train_labeled[i][1])
    # print(train_labeled[0][0].size(), train_labeled[0][1])

    # img = Image.fromarray(
    #     (255 * train_labeled[i][0]).numpy().astype(np.uint8), 'RGB')
    # img.show()
    # label = train_labeled[i][1]
    # print("label: ", train_labeled.id_to_str[int(label)])

    # hist, bins = np.histogram(train_labeled.targets,
    #                           bins=np.arange(len(train_labeled.str_labels)))
    # print("Train hist: {},\n bins: {}".format(hist, bins))
    # hist, bins = np.histogram(valid.targets,
    #                           bins=np.arange(len(valid.str_labels)))
    # print("Validation hist: {},\n bins: {}".format(hist, bins))

    # plt.figure()
    # plt.hist(train_labeled.targets, bins=np.arange(
    #     len(train_labeled.str_labels)))
    # plt.title("Histogram of class labels for train labeled data")
    # plt.savefig("train_hist.png")
    # plt.close()

    # plt.figure()
    # plt.hist(valid.targets, bins=np.arange(
    #     len(valid.str_labels)))
    # plt.title("Histogram of class labels for validation labeled data")
    # plt.savefig("valid_hist.png")
    # plt.close()
