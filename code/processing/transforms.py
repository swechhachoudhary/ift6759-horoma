import argparse
import os
import sys

import json
import pickle

import math
import random
from numbers import Number

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from torchvision.transforms.functional import to_tensor

parser = argparse.ArgumentParser()
parser.add_argument('--pickle_path', default=None, help="File containing the annotations")
parser.add_argument('--data_dir', default=None, help="Directory containing the dataset")


class ComposeMultipleInputs(object):
    """Compose multiple transforms with image and annotations as inputs."""

    def __init__(self, transforms):
        """
        Parameters
        ----------
        transforms : list
            List of transforms to be applied.
        """
        self.transforms = transforms

    def __call__(self, image, metadata):
        """Calls the class and iterates over al transforms applying them.

        Parameters
        ----------
        image : PIL image
            Image to be transformed.
        metadata : dict
            Dictionary containing annotations
        Returns
        -------
        image : PIL image
            Image with all transformations applied.
        metadata : dict
            Updated annotation dictionary.
        """
        for transform in self.transforms:
            image, metadata = transform(image, metadata)

        return image, metadata


class BoundingBoxCrop(object):
    """Crop the smallest box that encloses all the provided bounding boxes."""

    def __init__(self, margin=0.3):
        """
        Parameters
        ----------
        margin : Number
            Decimal to specify how much margin to add to the smallest box that encloses all bounding boxes.
        """
        if isinstance(margin, Number):
            self.margin = margin
        else:
            raise TypeError

    def __call__(self, image, metadata):
        """ Calls the class to perform the transform.

        Parameters
        ----------
        image : PIL image
            Image to be cropped.
        metadata : dict
            Dictionary containing annotations.

        Returns
        -------
        cropped_image
            Image cropped around the bounding boxes with the additional margin.
        metadata
            Updated metadata dictionary (bounding boxes coordinates change).
        """
        bounding_box = self.get_bounding_box(metadata)
        x, y, width, height = self.scale_bounding_box(bounding_box, image)
        cropped_image = self.crop(image, x, y, width, height)
        metadata = self.update_annotations(metadata, x, y)

        return cropped_image, metadata

    def scale_bounding_box(self, bounding_box, image):
        """Scales the bounding box in both x and y directions by the specified margin.

        Parameters
        ----------
        bounding_box : tuple
            X coordinate, y coordinate, width and height.
        image : PIL image
            Image to be transformed.

        Returns
        -------
        x_scaled : int
            x image coordinate scaled after adding the margin.
        y_scaled : int
            y image coordinate scaled after adding the margin.
        width_scaled : int
            width scaled after adding the margin.
        height_scaled : int
            height scaled after adding the margin.
        """
        x, y, width, height = bounding_box
        max_width, max_height = image.size

        x_scaled = max(int(x - (width * self.margin / 2)), 0)
        y_scaled = max(int(y - (height * self.margin / 2)), 0)
        width_scaled = min(math.ceil(x + width + (width * self.margin / 2)) - x_scaled, max_width)
        height_scaled = min(math.ceil(y + height + (height * self.margin / 2)) - y_scaled, max_height)

        return x_scaled, y_scaled, width_scaled, height_scaled

    @staticmethod
    def get_bounding_box(metadata):
        """Get the smallest box that encloses all the annotated bounding boxes.

        Parameters
        ----------
        metadata : dict
            Dictionary containing the metadata.

        Returns
        -------
        x : int
            Horizontal position of the top-left corner of the bounding box.
        y : int
            Vertical position of the top-left corner of the bounding box.
        width : int
            Horizontal length of the bounding box.
        height : int
            Vertical length of the bounding box.
        """
        x_coordinates = metadata.get('left')
        y_coordinates = metadata.get('top')
        widths = metadata.get('width')
        heights = metadata.get('height')

        x = min(x_coordinates)
        y = min(y_coordinates)
        width = - x + max([x + w for x, w in zip(x_coordinates, widths)])
        height = - y + max([y + h for y, h in zip(y_coordinates, heights)])

        return x, y, width, height

    @staticmethod
    def crop(image, x, y, width, height):
        """Crop an image using the top position (x, y), width and height.

        Parameters
        ----------
        image : PIL Image
            Image to be cropped.
        x : int
            Horizontal starting point for the crop.
        y : int
            Vertical starting point for the crop.
        width : int
            Horizontal length of the crop.
        height : int
            Vertical length of the crop.

        Returns
        -------
        cropped_image : PIL image
            Image cropped using specified dimensions.
        """
        # Add 1 to keep last pixel of the calculated (right, bottom) corner
        cropped_image = image.crop((x, y, x + width + 1, y + height + 1))

        return cropped_image

    @staticmethod
    def update_annotations(metadata, x, y):
        """Update the annotation dictionary.

        Parameters
        ----------
        metadata : dict
            Dictionary containing the metadata.
        x : int
            Horizontal starting point for the crop.
        y : int
            Vertical starting point for the crop.

        Returns
        -------
        metadata: dict
            Updated annotation dictionary.
        """
        metadata.update({
            'left': [x_coordinate - x for x_coordinate in metadata.get('left')],
            'top': [y_coordinate - y for y_coordinate in metadata.get('top')]
        })

        return metadata


class Rescale(object):
    """Rescale the image in to a given size and update annotations."""

    def __init__(self, output_size):
        """

        Parameters
        ----------
        output_size : tuple
            Tuple containing desired (width, height) of the output image.
        """
        assert isinstance(output_size, tuple)
        self.output_size = output_size

    def __call__(self, image, metadata):
        """ Calls the class to perform the transform.
        Parameters
        ----------
        image : PIL image
            Image to be transformed.
        metadata : dict
            Dictionary containing annotations

        Returns
        -------
        image : PIL image
            Image with all transformations applied.
        metadata : dict
            Updated annotation dictionary.
        """
        width, height = image.size

        new_height, new_width = self.output_size
        new_height, new_width = int(new_height), int(new_width)

        image = image.resize((new_height, new_width))
        metadata = self.update_annotations(metadata, new_width / width, new_height / height)

        return image, metadata

    @staticmethod
    def update_annotations(metadata, width_ratio, height_ratio):
        """Update the annotation dictionary according to the ratios between widths and heights.

        Parameters
        ----------
        metadata : dict
            Dictionary containing the metadata.
        width_ratio : float
            Ratio of change between the widths of scaled and original image.
        height_ratio : float
            Ratio of change between the heights of scaled and original image.

        Returns
        -------
        metadata: dict
            Updated annotation dictionary.
        """
        metadata.update({
            'left': [int(x_coordinate * width_ratio) for x_coordinate in metadata.get('left')],
            'top': [int(y_coordinate * height_ratio) for y_coordinate in metadata.get('top')],
            'width': [int(width * width_ratio) for width in metadata.get('width')],
            'height': [int(height * height_ratio) for height in metadata.get('height')],
        })

        return metadata


class RandomCrop(object):
    """Crop a random part smaller than the image."""

    def __init__(self, size=(54, 54)):
        """
        Parameters
        ----------
        size : tuple or int
            Specifies the dimension of the crop as tuple (width, height) or int (value, value).
        """
        if isinstance(size, tuple):
            self.size = size
        else:
            self.size = (int(size), int(size))

    def __call__(self, image, metadata):
        """ Calls the class to perform the transform.
        Parameters
        ----------
        image : PIL image
            Image to be transformed.
        metadata : dict
            Dictionary containing annotations

        Returns
        -------
        image : PIL image
            Image with all transformations applied.
        metadata : dict
            Updated annotation dictionary.
        """
        x, y, width, height = self.get_params(image, self.size)
        image = self.crop(image, x, y, width, height)
        metadata = self.update_annotations(metadata, x, y)
        return image, metadata

    def update_annotations(self, metadata, x_crop, y_crop):
        """Update the annotation dictionary according to the random crop done (only uses x and y position of crop).

        Parameters
        ----------
        metadata : dict
            Dictionary containing the metadata.
        x_crop : int
            Horizontal starting point for the crop.
        y_crop : int
            Vertical starting point for the crop.

        Returns
        -------
        metadata: dict
            Updated annotation dictionary.
        """
        x_update, width_update = [], []
        for x, width in zip(metadata.get('left'), metadata.get('width')):
            x_difference = x - x_crop
            x_update.append(max(0, x_difference))
            width_update.append(min(self.size[0]-1, x_difference + width)-x_update[-1])

        y_update, height_update = [], []
        for y, height in zip(metadata.get('top'), metadata.get('height')):
            y_difference = y - y_crop
            y_update.append(max(0, y_difference))
            height_update.append(min(self.size[-1]-1, y_difference + height) - y_update[-1])

        metadata.update({
            'left': x_update,
            'top': y_update,
            'width': width_update,
            'height': height_update,
        })

        return metadata

    @staticmethod
    def get_params(image, output_size):
        """Get parameters for ``crop`` for a random crop.

        Args:
            image (PIL Image): Image to be cropped.
            output_size (tuple): Expected output size of the crop.

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        width, height = image.size
        crop_width, crop_height = output_size
        if width == crop_width and height == crop_height:
            return 0, 0, width, height
        assert (width - crop_width) >= 0 and (height - crop_height) >= 0
        x = random.randint(0, width - crop_width)
        y = random.randint(0, height - crop_height)

        return x, y, crop_width, crop_height

    @staticmethod
    def crop(image, x, y, width, height):
        """Crop an image using the top position (x, y), width and height.

        Parameters
        ----------
        image : PIL Image
            Image to be cropped.
        x : int
            Horizontal starting point for the crop.
        y : int
            Vertical starting point for the crop.
        width : int
            Horizontal length of the crop.
        height : int
            Vertical length of the crop.

        Returns
        -------
        cropped_image : PIL image
            Image cropped using specified dimensions.
        """
        cropped_image = image.crop((x, y, x + width, y + height))

        return cropped_image


class ToTensor(object):
    """Convert a PIL image or ndarray to tensor."""

    def __call__(self, image, metadata):
        """

        Parameters
        ----------
        image : PIL image
            Image to be transformed to tensor.
        metadata : dict
            Annotation dictionary.

        Returns
        -------
        image : Tensor
            Converted image.
        metadata : dict
            Annotation dictionary with no modifications.
        """
        return to_tensor(image), metadata


if __name__ == '__main__':
    """Test for the transforms module, load some images, perform transformations and save them."""
    args = parser.parse_args()
    PKL_PATH = args.pickle_path
    IMG_DIR = args.data_dir

    assert  PKL_PATH is not None and IMG_DIR is not None, "Specify pickle path and image directory"

    print("Start test process", file=sys.stderr)

    with open(PKL_PATH, 'rb') as annotation_file:
        annotations = pickle.load(annotation_file)
    transform = ComposeMultipleInputs([
        BoundingBoxCrop(),
        Rescale((64, 64)),
        RandomCrop(),
    ])

    print('Create result directory')
    os.mkdir('figs')

    print("Initialized transforms", file=sys.stderr)

    for i in range(0, 10):
        print("Start reading process for image {}".format(i), file=sys.stderr)
        image_dictionary = annotations.get(i)
        filename = image_dictionary.get('filename')
        metadata = image_dictionary.get('metadata')
        image = Image.open(os.path.join(IMG_DIR, filename))
        img, metadata = transform(image, metadata)

        with open(os.path.join('figs', '{}.txt'.format(i)), 'w') as file:
            file.write(json.dumps(metadata))

        top = metadata.get('top')
        left = metadata.get('left')
        height = metadata.get('height')
        width = metadata.get('width')

        fig, ax = plt.subplots(1)
        ax.imshow(img)
        for idx, label in enumerate(metadata.get('label')):
            rect = patches.Rectangle((left[idx], top[idx]), width[idx], height[idx],
                                     linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)

        fig.savefig('figs/{}.png'.format(i))  # save the figure to file
        plt.close(fig)
        print("Finished reading process for image {}".format(i), file=sys.stderr)
