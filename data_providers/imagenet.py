'''we should use python3 for this file'''
import tempfile
import os
import pickle
import random

import numpy as np


from .base_provider import ImagesDataSet, DataProvider
from .downloader import download_data_url


def augment_image(image, pad):
    """Perform zero padding, randomly crop image to original size,
    maybe mirror horizontally"""
    init_shape = image.shape
    new_shape = [init_shape[0] + pad * 2,
                 init_shape[1] + pad * 2,
                 init_shape[2]]
    zeros_padded = np.zeros(new_shape)
    zeros_padded[pad:init_shape[0] + pad, pad:init_shape[1] + pad, :] = image
    # randomly crop to original size
    init_x = np.random.randint(0, pad * 2)
    init_y = np.random.randint(0, pad * 2)
    cropped = zeros_padded[
        init_x: init_x + init_shape[0],
        init_y: init_y + init_shape[1],
        :]
    flip = random.getrandbits(1)
    if flip:
        cropped = cropped[:, ::-1, :]
    return cropped


def augment_all_images(initial_images, pad):
    new_images = np.zeros(initial_images.shape)
    for i in range(initial_images.shape[0]):
        new_images[i] = augment_image(initial_images[i], pad=4)
    return new_images


class ImagenetDataSet(ImagesDataSet):
    def __init__(self, images, labels, n_classes=1000, shuffle, normalization,
                 augmentation):
        if shuffle is None:
            self.shuffle_every_epoch = False
        elif shuffle == 'once_prior_train':
            self.shuffle_every_epoch = False
            images, labels = self.shuffle_images_and_labels(images, labels)
        elif shuffle == 'every_epoch':
            self.shuffle_every_epoch = True
        else:
            raise Exception("Unknown type of shuffling")

        self.images = images
        self.labels = labels
        self.n_classes = n_classes
        self.augmentation = augmentation
        self.normalization = normalization
        self.images = self.normalize_images(images, self.normalization)
        self.start_new_epoch()

    def start_new_epoch(self):
        self._batch_counter = 0
        if self.shuffle_every_epoch:
            images, labels = self.shuffle_images_and_labels(
                self.images, self.labels)
        else:
            images, labels = self.images, self.labels
        if self.augmentation:
            images = augment_all_images(images, pad=4)
        self.epoch_images = images
        self.epoch_labels = labels

    @property
    def num_examples(self):
        return self.labels.shape[0]

    def next_batch(self, batch_size):
        start = self._batch_counter * batch_size
        end = (self._batch_counter + 1) * batch_size
        self._batch_counter += 1
        images_slice = self.epoch_images[start: end]
        labels_slice = self.epoch_labels[start: end]
        if images_slice.shape[0] != batch_size:
            self.start_new_epoch()
            return self.next_batch(batch_size)
        else:
            return images_slice, labels_slice


class ImagenetDataProvider(DataProvider):
    """Abstract class for imagenet readers"""

    def __init__(self, save_path=None, validation_set=None,
                 validation_split=None, shuffle=None, normalization=None,
                 one_hot=False, **kwargs):
                 
        self._save_path = save_path
        self.one_hot = one_hot

        train_fnames, test_fnames = self.get_filenames(self.save_path)

        # add train and validations datasets
        images, labels = self.read_imagenet(train_fnames)
        if validation_set is not None and validation_split is not None:
            split_idx = int(images.shape[0] * (1 - validation_split))
            self.train = ImagenetDataSet(
                images=images[:split_idx], labels=labels[:split_idx],
                n_classes=self.n_classes, shuffle=shuffle,
                normalization=normalization,
                augmentation=self.data_augmentation)
            self.validation = ImagenetDataSet(
                images=images[split_idx:], labels=labels[split_idx:],
                n_classes=self.n_classes, shuffle=shuffle,
                normalization=normalization,
                augmentation=self.data_augmentation)
        else:
            self.train = ImagenetDataSet(
                images=images, labels=labels,
                n_classes=self.n_classes, shuffle=shuffle,
                normalization=normalization,
                augmentation=self.data_augmentation)

        # add test set
        images, labels = self.read_imagenet(test_fnames)
        self.test = ImagenetDataSet(
            images=images, labels=labels,
            shuffle=None, n_classes=self.n_classes,
            normalization=normalization,
            augmentation=False)

        if validation_set and not validation_split:
            self.validation = self.test

    @property
    def save_path(self):
        if self._save_path is None:
            self._save_path = os.path.join('imagenet')
        return self._save_path

    @property
    def data_shape(self):
        return (32, 32, 3)

    @property
    def n_classes(self):
        return self._n_classes

    def get_filenames(self, save_path):
        """Return two lists of train and test filenames for dataset"""
        train_filenames = [
            os.path.join(
                save_path,
                'train_data_batch_%d' % i) for i in range(1, 11)]
        test_filenames = [os.path.join(save_path, 'val_data')]
        return train_filenames, test_filenames

    def read_imagenet(self, filenames):
        images_res = []
        labels_res = []
        for fname in filenames:
            with open(fname, 'rb') as f:
                images_and_labels = pickle.load(f)
            images = images_and_labels['data']
            images = np.dstack((images[:, :1024], images[:, 1024:2048], images[:, 2048:]))
            images = images.reshape((x.shape[0], 32, 32, 3))
            images_res.append(images)
            labels_res.append(images_and_labels['labels'])
        images_res = np.vstack(images_res)
        labels_res = np.hstack(labels_res)
        if self.one_hot:
            labels_res = self.labels_to_one_hot(labels_res)   
        return images_res, labels_res

class ImagenetAugmentedDataProvider(ImagenetDataProvider):
    data_augmentation = True