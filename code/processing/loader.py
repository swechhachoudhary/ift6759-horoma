import os
import sys
import pickle

import numpy as np
from PIL import Image

import torch
from torchvision.transforms import ToPILImage, Normalize
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler

from processing.transforms import ComposeMultipleInputs, BoundingBoxCrop, Rescale, RandomCrop, ToTensor

# Imported for testing
from utils.classes import Params

# Transforms to be used when defining loaders
train_transform = ComposeMultipleInputs([
    BoundingBoxCrop(),
    Rescale((64, 64)),
    RandomCrop((54, 54)),
    Rescale((224, 224)),
    ToTensor(),
])

eval_transform = ComposeMultipleInputs([
    BoundingBoxCrop(),
    Rescale((64, 64)),
    RandomCrop((54, 54)),
    Rescale((224, 224)),
    ToTensor(),
])

normalize_transform = Normalize((0.434, 0.442, 0.473), (0.2, 0.202, 0.198))


class SVHNDataset(Dataset):
    """Street View House Numbers dataset."""

    def __init__(self, pickle_path, image_dir, transform=None, normalize=True):
        """

        Parameters
        ----------
        pickle_path : str
            Path containing the pickle file with filename and metadata.
        image_dir : str
            Path to the directory containing the images.
        transform :
            Instance of a class performing multiple or single transforms to the data.
        normalize : bool
            Normalize the images or not.
        """
        super()
        with open(pickle_path, 'rb') as annotation_file:
            self.annotations = pickle.load(annotation_file)
        self.img_dir = image_dir
        self.transform = transform
        self.normalize = normalize

    def __len__(self):
        """Returns the length of the defined dataset"""
        return len(self.annotations)

    def __getitem__(self, idx):
        """Get a sample from the dataset consisting of (image, metadata)

        Parameters
        ----------
        idx : int
            Index of the desired sample to retrieve.
        Returns
        -------
        image : torch.Tensor
            Tensor representing the image with the pipeline of transforms applied.
        label :
            Label desired to be used in the training phase. Can be modified using the label_modifier function.
        """
        image_dictionary = self.annotations.get(idx)
        filename = image_dictionary['filename']
        metadata = image_dictionary['metadata']
        image = Image.open(os.path.join(self.img_dir, filename)).convert('RGB')

        if self.transform:
            image, metadata = self.transform(image, metadata)

            if self.normalize:
                image = normalize_transform(image)

        metadata.update({
            'height': np.asarray(metadata['height']),
            'width': np.asarray(metadata['width']),
            'left': np.asarray(metadata['left']),
            'top': np.asarray(metadata['top']),
            'label': np.asarray(metadata['label'])
        })

        # Modify label modifier function depending on block task
        return image, self.label_modifier(metadata)

    @staticmethod
    def label_modifier(metadata):
        """Returns the number of digits in the image. 5 or more digits are grouped in one class.

        Note:
        This function is the one to be modified depending on the task to be addressed in the block.

        Parameters
        ----------
        metadata : dict
            Dictionary containing annotations.

        Returns
        -------
        digits : int
            Number of digits in the image.
        """

        return min(len(metadata.get('label')), 5) - 1


def fetch_dataloader(type, params, pickle_path, image_dir):
    """Fetches the dataloader for the desired data split.

    Parameters
    ----------
    type : str
        Split of the data to get the data loader for.
    params : object
        Params object containing configuration information.
    pickle_path : dir
        Path containing the pickle file with filename and metadata.
    image_dir : str
        Path to the directory containing the Images.
    Returns
    -------
    dataloaders : dict
        Dictionary containing the dataloaders fot the desired split.
    """
    assert type in ['train', 'test'], "Invalid dataloader type"
    assert ((params.ratio >= 0) and (params.ratio <= 1)), "Split must be between 0 and 1"

    dataloaders = {}

    # Return test data loader is split is test (using eval_transformer) else return dictionary containing data loaders
    # for the train and validation sets.
    if type == 'test':
        test_dataloader = DataLoader(SVHNDataset(pickle_path, image_dir, eval_transform),
                                     batch_size=params.batch_size, shuffle=False,
                                     num_workers=params.num_workers, pin_memory=params.cuda)

        dataloaders.update({
            'test': test_dataloader
        })

    else:
        train_dataset = SVHNDataset(pickle_path, image_dir, train_transform)

        length_train = len(train_dataset)
        indices = list(range(length_train))
        split = int(params.ratio * length_train)

        if params.shuffle:
            np.random.seed(params.random_seed)
            np.random.shuffle(indices)

        train_indices, val_indices = indices[split:], indices[:split]
        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler = SubsetRandomSampler(val_indices)

        train_dataloader = DataLoader(train_dataset, batch_size=params.batch_size, sampler=train_sampler,
                                      num_workers=params.num_workers, pin_memory=params.cuda)
        val_dataloader = DataLoader(train_dataset, batch_size=params.batch_size, sampler=val_sampler,
                                    num_workers=params.num_workers, pin_memory=params.cuda)

        dataloaders.update({
            'train': train_dataloader,
            'val': val_dataloader
        })

    return dataloaders


if __name__ == "__main__":
    """Simple test to call the data loaders and obtain some image plots and general statistics (mean and std)."""

    print("Starting module test", file=sys.stderr)
    json_path = os.path.join('experiments', 'base_model', 'params.json')
    params = Params(json_path)
    params.ratio = 0.001

    pickle_train = os.path.join('..', 'digit-detection', 'data', 'SVHN', 'train_metadata.pkl')
    images_train = os.path.join('..', 'digit-detection', 'data', 'SVHN', 'train')
    train_dl = fetch_dataloader('train', params, pickle_train, images_train)

    r, g, b, n = 0, 0, 0, 0

    for i, (train_batch, _) in enumerate(train_dl.get('train')):
        rgb = train_batch.transpose(0, 1).contiguous().view(train_batch.transpose(0, 1).shape[0], -1).mean(1)
        bs = train_batch.shape[0] * train_batch.shape[2] * train_batch.shape[3]
        r += rgb[0] * bs
        g += rgb[1] * bs
        b += rgb[2] * bs
        n += bs

    R_mean = torch.div(r, n)
    G_mean = torch.div(g, n)
    B_mean = torch.div(b, n)

    print("R pixel mean: {}".format(R_mean))
    print("G pixel mean: {}".format(G_mean))
    print("B pixel mean: {}".format(B_mean))

    r_std, g_std, b_std, n = 0, 0, 0, 0

    for i, (train_batch, _) in enumerate(train_dl.get('train')):
        rgb = train_batch.transpose(0, 1).contiguous().view(train_batch.transpose(0, 1).shape[0], -1).var(1)
        bs = train_batch.shape[0] * train_batch.shape[2] * train_batch.shape[3]
        r_std += rgb[0] * bs
        g_std += rgb[1] * bs
        b_std += rgb[2] * bs
        n += bs

    r_std = torch.div(r_std, n) ** 0.5
    g_std = torch.div(g_std, n) ** 0.5
    b_std = torch.div(b_std, n) ** 0.5

    print("R pixel std: {}".format(r_std))
    print("G pixel std: {}".format(g_std))
    print("B pixel std: {}".format(b_std))