from __future__ import division
import torch
import random
import numpy as np
from scipy.misc import imresize
from scipy.ndimage.interpolation import zoom
import warnings
import imgaug as ia
from imgaug import augmenters as iaa

'''Set of tranform random routines that takes list of inputs as arguments,
in order to have random but coherent transformations.'''


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, images, depth, intrinsics):
        for t in self.transforms:
            images, depth, intrinsics = t(images, depth, intrinsics)
        return images, depth, intrinsics


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, images, depth, intrinsics):
        for tensor in images:
            for t, m, s in zip(tensor, self.mean, self.std):
                t.sub_(m).div_(s)
        return images, depth, intrinsics


class ArrayToTensor(object):
    """Converts a list of numpy.ndarray (H x W x C) along with a intrinsics matrix to a list of torch.FloatTensor of shape (C x H x W) with a intrinsics tensor."""

    def __call__(self, images, depth, intrinsics):
        tensors = []
        for im in images:
            # put it from HWC to CHW format
            im = np.transpose(im, (2, 0, 1))
            # handle numpy array
            tensors.append(torch.from_numpy(im).float()/255)
        return tensors, depth, intrinsics


class RandomScaleCrop(object):
    """Randomly zooms images up to 15% and crop them to keep same size as before."""

    def __call__(self, images, depth, intrinsics):
        assert intrinsics is not None
        output_intrinsics = np.copy(intrinsics)

        out_h = 240
        out_w = 320
        in_h, in_w, _ = images[0].shape
        x_scaling = np.random.uniform(out_w/in_w, 1)
        y_scaling = np.random.uniform(out_h/in_h, 1)
        scaled_h, scaled_w = round(in_h * y_scaling), round(in_w * x_scaling)

        output_intrinsics[0] *= x_scaling
        output_intrinsics[1] *= y_scaling
        scaled_images = [imresize(im, (scaled_h, scaled_w)) for im in images]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scaled_depth = zoom(depth, (y_scaling, x_scaling))

        offset_y = np.random.randint(scaled_h - out_h + 1)
        offset_x = np.random.randint(scaled_w - out_w + 1)
        cropped_images = [im[offset_y:offset_y + out_h, offset_x:offset_x + out_w, :] for im in scaled_images]
        cropped_depth = scaled_depth[offset_y:offset_y + out_h, offset_x:offset_x + out_w]

        output_intrinsics[0,2] -= offset_x
        output_intrinsics[1,2] -= offset_y

        return cropped_images, cropped_depth, output_intrinsics


class ColorJitter(object):
    """Randomly change image color or contrast"""

    def __init__(self):
        self.some_aug = iaa.SomeOf(
            (0, 2),
            [
                iaa.AdditiveGaussianNoise(
                    loc=0,
                    scale=(0.0,
                           0.01 * 255)),  # add gaussian noise to images
                iaa.ContrastNormalization(
                    (0.8, 1.2),
                    per_channel=0.25),  # improve or worsen the contrast
                iaa.Multiply((0.8, 1.2), per_channel=0.25),
                iaa.Add((-25, 25), per_channel=0.25)
            ],
            random_order=True)

    def __call__(self, images, depth, intrinsics):

        out_images = [self.some_aug.augment_image(it) for it in images]
        return out_images, depth, intrinsics