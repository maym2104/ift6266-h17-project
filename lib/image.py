"""
Copyright (c) 2017 - Philip Paquette

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import os
import skimage.color as color
import skimage.io as io
import skimage.transform as transform
import time
import theano
import warnings
from settings import SAVED_DIR, MAX_WIDTH, MAX_HEIGHT, AUTO_RESIZE, DATASET_AUGMENTATION
from .utils import mkdir_p

class Image(object):
    """Image manipulation class

    This class loads images from disk and perform manipulation on them
    (e.g. masking, rescaling)
    """
    def __init__(self, dataset_image=None):
        if dataset_image is not None:
            self.full_path = dataset_image['full_path']
            self.id = dataset_image['id']
            self._load_from_disk()
            if not DATASET_AUGMENTATION:
                process_fn = self.resize if AUTO_RESIZE else self.crop
                process_fn()

    def downscale(self, factor):
        """ Downscale the image with factor (Uses mean pooling) """
        self.data = self._to_theano(transform.downscale_local_mean(self._from_theano(self.data), (1, factor, factor)))

    def upscale(self, factor, use_smoothing=False):
        """ Upscale the non-masked, masked, and mask with factor (with optional Gaussian filter for smoothing) """
        scale_fn = transform.pyramid_expand if use_smoothing else transform.rescale
        self.data = self._to_theano(scale_fn(self._from_theano(self.data), factor))

    def crop(self, offset_h=None, offset_v=None, width=None, height=None):
        """ Crop image """
        offset_h = (self.width - MAX_WIDTH) // 2 if offset_h is None else offset_h
        offset_v = (self.height - MAX_HEIGHT) // 2 if offset_v is None else offset_v
        target_width = MAX_WIDTH if width is None else width
        target_height = MAX_HEIGHT if height is None else height
        self.data = self.data[:, offset_v:target_height + offset_v, offset_h:target_width + offset_h]

    def resize(self, target_width=MAX_WIDTH, target_height=MAX_HEIGHT, keep_ratio=False):
        """" Resizes the image to the target dimension """
        if not keep_ratio:
            self.data = self._to_theano(transform.resize(self._from_theano(self.data), (target_height, target_width, 3)))
            self.height, self.width = target_height, target_width
        else:
            scale = float(target_height) / min(self.height, self.width)
            self.height, self.width = int(round(scale * self.height)),  int(round(scale * self.width))
            self.data = self._to_theano(transform.resize(self._from_theano(self.data), (self.height, self.width, 3)))

            offset_h, offset_v = int((self.width - target_width) / 2.), int((self.height - target_height) / 2.)
            self.crop(offset_h, offset_v, target_width, target_height)
            self.height, self.width = target_height, target_width

    def flip(self):
        self.data = np.flip(self.data, axis=2)

    def rotate(self, angle):
        self.data = self._to_theano(transform.rotate(self._from_theano(self.data), -1. * angle, resize=False, mode='edge'))

    def to_gray(self):
        self.data = self._to_theano(color.gray2rgb(color.rgb2gray(self._from_theano(self.data))))

    def save(self, path_prefix = '', file_prefix = '', image_data = None):
        """ Save image to disk """
        image_data = self.data if image_data is None else image_data
        timestamp = int(time.time() * 1000)
        target_folder = os.path.join(SAVED_DIR, path_prefix)
        if not os.path.exists(target_folder):
            mkdir_p(target_folder)
        file_name = os.path.join(target_folder, 'image%s-%d.jpg' % (file_prefix, timestamp))
        io.imsave(file_name, np.array(255. * self._from_theano(image_data)).astype('uint8'))

    def display(self, image_data = None):
        """ Display masked image """
        plt.figure()
        plt.axis('off')
        plt.imshow(self._from_theano(image_data if image_data is not None else self.data))
        plt.show()

    # -----------------------
    # --- Private Methods ---
    # -----------------------
    def _load_from_disk(self, target_dims=None):
        """ Load the image file from disk """
        if not os.path.exists(self.full_path):
            warnings.warn('Unable to load image - Path: %s.' % (self.full_path))
            return
        data_from_disk = np.array(io.imread(self.full_path), dtype=theano.config.floatX) / 255.
        if (len(data_from_disk.shape) == 2):
            data_from_disk = color.gray2rgb(data_from_disk)
        self.data = self._to_theano(data_from_disk)
        self.height = self.data.shape[1]
        self.width = self.data.shape[2]

    def _to_theano(self, target):
        """ Converts numpy array from (height, width, channel) to (channel, height, width) """
        return np.transpose(target, (2, 0, 1))

    def _from_theano(self, target):
        """ Converts numpy array from (channel, height, width) to (height, width, channel) """
        return np.transpose(target, (1, 2, 0))

