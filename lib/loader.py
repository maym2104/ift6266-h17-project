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
from .pycocotools.coco import COCO
import pickle as cPickle
import os
import numpy as np
import Queue as queue
import sys
import theano
import time
from .utils import floatX, mkdir_p
from threading import Thread
from settings import DATA_DIR, MAX_ITEMS_PER_DATASET, LOAD_DATASET_IN_MEMORY, MAX_HEIGHT, MAX_WIDTH, SAVED_DIR, DATASET_AUGMENTATION
from settings import MAX_TRAINING_ITEMS, MAX_VALIDATION_ITEMS, MAX_TEST_ITEMS
from .image import Image
from .rng import py_rng

MAX_ITEMS_FOR_PICKLE = 10000

# Folder structure

# / ROOT / coco / annotations / captions_train2014.json
# / ROOT / coco / annotations / captions_val2014.json
# / ROOT / coco / images / train2014 / <images>
# / ROOT / coco / images / val2014 / <images>
# / ROOT / saved / dataset.pkl

class Loader(object):
    """Dataset loader

    This class loads the dataset from disk and splits it into a training,
    validation, and test set.
    """
    def __init__(self, hparams):
        self.train = None
        self.valid = None
        self.test = None
        self.hparams = hparams
        self.nb_examples = {'train': 0, 'valid': 0, 'test': 0}
        self.ix_last_items = {'train': 0, 'valid': 0, 'test': 0}
        self.dataset = {'train': None, 'valid': None, 'test': None}
        self.buffer = {'train': queue.Queue(), 'valid': queue.Queue(), 'test': queue.Queue()}

    def load_dataset(self):
        """Load the dataset and splits into training, validation and testing"""
        if DATASET_AUGMENTATION or not LOAD_DATASET_IN_MEMORY:
            train_coco, valid_coco = self._load_coco_dataset()
            coco_dataset = train_coco.dataset['images'] + valid_coco.dataset['images']
            options = {
                'len_train_coco': len(train_coco.dataset['images']),
                'train_coco': train_coco,
                'valid_coco': valid_coco
            }
            dataset, self.train, self.valid, self.test = self._build_disk_dataset(coco_dataset, options)

        else:
            dataset_file = os.path.join(SAVED_DIR, 'dataset.pkl')
            if os.path.exists(dataset_file):
                # Loading pickled dataset
                with open(dataset_file, 'rb') as f:
                    split_dataset = cPickle.load(f)
                options = {}
                pickled_dataset = {
                    'data': np.concatenate([split_dataset[i]['data'] for i in range(len(split_dataset))]),
                    'captions': []
                }
                for i in range(len(split_dataset)):
                    pickled_dataset['captions'] = pickled_dataset['captions'] + split_dataset[i]['captions']
            else:
                # Building pickled dataset
                train_coco, valid_coco = self._load_coco_dataset()
                coco_dataset = train_coco.dataset['images'] + valid_coco.dataset['images']
                options = {
                    'len_train_coco': len(train_coco.dataset['images']),
                    'nb_images': len(train_coco.dataset['images']) + len(valid_coco.dataset['images']),
                    'train_coco': train_coco,
                    'valid_coco': valid_coco,
                    'dataset_file': dataset_file
                }
                pickled_dataset = self._pickle_dataset(coco_dataset, options)
            dataset, self.train, self.valid, self.test = self._build_pickled_dataset(pickled_dataset, options)

        self.nb_examples['train'] = len(self.train['captions'])
        self.nb_examples['valid'] = len(self.valid['captions'])
        self.nb_examples['test']  = len(self.test['captions'])
        self.dataset = {'train': self.train, 'valid': self.valid, 'test': self.test}

    def start_new_epoch(self):
        """ Resets the number of processed items for train, valid, and test and clear the buffer"""
        self.ix_last_items = {'train': 0, 'valid': 0, 'test': 0}
        for dataset_name in ['train', 'valid', 'test']:
            while self.buffer[dataset_name].qsize() > 0:
                _ = self.buffer[dataset_name].get()

    def get_next_batch(self, batch_size, dataset_name, img_scale='sigmoid', skip_images=False, print_status=False, nb_threads=4):
        """ Returns the next batch of images and captions using multiple threads """
        assert dataset_name in ['train', 'valid', 'test']
        assert img_scale in ['sigmoid', 'tanh']
        nb_to_process = min(batch_size, self.nb_examples[dataset_name] - self.ix_last_items[dataset_name])
        if nb_to_process == 0:
            return None, None, None, None, True
        batch_data = np.zeros((nb_to_process, 3, MAX_HEIGHT, MAX_WIDTH), dtype=theano.config.floatX)
        batch_real_captions = []
        batch_matching_captions = []
        batch_fake_captions = []
        last_batch_in_epoch = True if self.ix_last_items[dataset_name] + batch_size >= self.nb_examples[dataset_name] else False

        # Generating
        if print_status and DATASET_AUGMENTATION:
            print('.... Retrieving next batch of %d examples' % (nb_to_process))
        if not DATASET_AUGMENTATION:
            for ix in range(nb_to_process):
                fake_ix = py_rng.randint(0, self.nb_examples[dataset_name] - 1)

                if not DATASET_AUGMENTATION and LOAD_DATASET_IN_MEMORY:
                    # Loading from pickled dataset
                    curr_ix = ix + self.ix_last_items[dataset_name]
                    if fake_ix == curr_ix:
                        fake_ix = (fake_ix + 1) % self.nb_examples[dataset_name]
                    if not skip_images:
                        batch_data[ix, :, :, :] = self.dataset[dataset_name]['data'][curr_ix]
                    py_rng.shuffle(self.dataset[dataset_name]['captions'][curr_ix])
                    batch_real_captions.append(self.dataset[dataset_name]['captions'][curr_ix][0])
                    batch_matching_captions.append(self.dataset[dataset_name]['captions'][curr_ix][1])
                    batch_fake_captions.append(py_rng.choice(self.dataset[dataset_name]['captions'][fake_ix]))

                else:
                    # Loading from disk
                    curr_ix = ix + self.ix_last_items[dataset_name]
                    if fake_ix == curr_ix:
                        fake_ix = (fake_ix + 1) % self.nb_examples[dataset_name]
                    if not skip_images:
                        img = Image(self.dataset[dataset_name]['data'][curr_ix])
                        batch_data[ix, :, :, :] = img.data
                    py_rng.shuffle(self.dataset[dataset_name]['captions'][curr_ix])
                    batch_real_captions.append(self.dataset[dataset_name]['captions'][curr_ix][0])
                    batch_matching_captions.append(self.dataset[dataset_name]['captions'][curr_ix][1])
                    batch_fake_captions.append(py_rng.choice(self.dataset[dataset_name]['captions'][fake_ix]))
        else:
            # Generating through threads
            nb_samples = [nb_to_process // nb_threads] * nb_threads
            nb_samples[-1] += nb_to_process - sum(nb_samples)
            threads = [None] * nb_threads
            results = [None] * nb_threads

            for i in range(len(threads)):
                threads[i] = Thread(target=self._generate_samples_worker, args=(nb_samples, dataset_name, skip_images, results, i))
                threads[i].start()
            for i in range(len(threads)):
                threads[i].join()

            # Merging
            batch_data = np.concatenate([result[0] for result in results], axis=0)
            batch_real_captions = [item for sublist in [result[1] for result in results] for item in sublist]
            batch_matching_captions = [item for sublist in [result[2] for result in results] for item in sublist]
            batch_fake_captions = [item for sublist in [result[3] for result in results] for item in sublist]

        if img_scale == 'tanh':
            batch_data = batch_data * 2. - 1.
        if print_status:
            print('.... Done retrieving next batch of %d examples' % (nb_to_process))
        self.ix_last_items[dataset_name] += nb_to_process
        return floatX(batch_data), batch_real_captions, batch_matching_captions, batch_fake_captions, last_batch_in_epoch

    def fill_buffer(self, batch_size, img_scale='sigmoid'):
        """ Fills a buffer of train, valid, and test batches in the background """
        assert img_scale in ['sigmoid', 'tanh']
        threads = [None] * 3
        for ix, dataset_name in enumerate(['train', 'valid', 'test']):
            threads[ix] = Thread(target=self._fill_buffer_worker, args=(batch_size, dataset_name, img_scale))
            threads[ix].daemon = True
            threads[ix].start()


    # -----------------------
    # --- Private Methods ---
    # -----------------------
    def _load_coco_dataset(self):
        """Loads a COCO dataset (captions)
        Returns the loaded training and validation COCO object
        """
		
        #captions_path = os.path.join(os.path.join(DATA_DIR, 'annotations'), 'dict_key_imgID_value_caps_train_and_valid.pkl')
        #with open(captions_path, 'rb') as f:
        #    captions = cPickle.load(f)
        #train_coco = captions['train']
        #valid_coco = captions['valid']
		
        print('..... Initializing COCO')
        with open(os.devnull, 'w') as f:
            old_stdout, sys.stdout = sys.stdout, f
            train_coco = COCO('%s/annotations/captions_train2014.json' % (DATA_DIR))
            valid_coco = COCO('%s/annotations/captions_val2014.json' % (DATA_DIR))
            sys.stdout = old_stdout
        print('..... Done initializing COCO')
        return train_coco, valid_coco

    def _get_image_full_path(self, i, options, file_name):
        """ Returns the image full path """
        (curr_coco, curr_img_dir) = (options['train_coco'], 'train2014') if i < options['len_train_coco'] else (options['valid_coco'], 'val2014')
        return os.path.join(DATA_DIR, 'images', curr_img_dir, file_name)

    def _get_captions(self, i, options, id):
        """ Returns the captions for an image id """
        (curr_coco, curr_img_dir) = (options['train_coco'], 'train2014') if i < options['len_train_coco'] else (options['valid_coco'], 'val2014')
        return [caption['caption'] for caption in curr_coco.loadAnns(curr_coco.getAnnIds(imgIds=id))]

    def _get_dataset_length(self, dataset):
        """ Returns the nb of images in train, valid, test given a loaded dataset """
        perc_train = self.hparams.get('perc_train', 0.7)
        perc_valid = self.hparams.get('perc_valid', 0.15)
        perc_test = self.hparams.get('perc_test', 0.15)

        nb_images = len(dataset) if not 'captions' in dataset else len(dataset['captions'])
        nb_train = int(perc_train / (perc_train + perc_valid + perc_test) * nb_images)
        nb_valid = int(perc_valid / (perc_train + perc_valid + perc_test) * nb_images)
        nb_test = nb_images - nb_train - nb_valid

        if MAX_ITEMS_PER_DATASET != -1:
            nb_train = min(MAX_ITEMS_PER_DATASET, nb_train)
            nb_valid = min(MAX_ITEMS_PER_DATASET, nb_valid)
            nb_test = min(MAX_ITEMS_PER_DATASET, nb_test)
        nb_train = nb_train if MAX_TRAINING_ITEMS == -1 else min(MAX_TRAINING_ITEMS, nb_train)
        nb_valid = nb_valid if MAX_VALIDATION_ITEMS == -1 else min(MAX_VALIDATION_ITEMS, nb_valid)
        nb_test = nb_test if MAX_TEST_ITEMS == -1 else min(MAX_TEST_ITEMS, nb_test)
        nb_images = nb_train + nb_valid + nb_test

        return nb_images, nb_train, nb_valid, nb_test

    def _pickle_dataset(self, coco_dataset, options):
        """ Pickle the full dataset to disk (with images resized) -- Does not support dataset augmentation """
        nb_images = options['nb_images']
        nb_split = nb_images // MAX_ITEMS_FOR_PICKLE
        if nb_images > nb_split * MAX_ITEMS_FOR_PICKLE:
            nb_split += 1
        pickled_dataset = {
            'data': np.zeros((nb_images, 3, MAX_HEIGHT, MAX_WIDTH), dtype=theano.config.floatX),
            'captions': []
        }

        for i in range(nb_images):
            if i % 100 == 99:
                print('..... Loaded %d of %d images in memory' % (i+1, nb_images))
            curr_image = coco_dataset[i]
            curr_captions = self._get_captions(i, options, curr_image['id'])
            curr_image['full_path'] = self._get_image_full_path(i, options, curr_image['file_name'])
            image_data = Image(curr_image).data
            pickled_dataset['data'][i, :, :, :] = image_data
            pickled_dataset['captions'].append(curr_captions)
        print('... Done loading images in memory')

        if not os.path.exists(os.path.dirname(options['dataset_file'])):
            mkdir_p(os.path.dirname(options['dataset_file']))

        # Splitting into sub arrays to avoid numpy error with large arrays
        split_dataset = []
        for ix in range(nb_split):
            split_dataset.append({
                'data': pickled_dataset['data'][(ix) * MAX_ITEMS_FOR_PICKLE: (ix+1) * MAX_ITEMS_FOR_PICKLE, :, :, :],
                'captions': pickled_dataset['captions'][(ix) * MAX_ITEMS_FOR_PICKLE: (ix+1) * MAX_ITEMS_FOR_PICKLE]})
        with open(options['dataset_file'], 'wb') as f:
            cPickle.dump(split_dataset, f, 2)
        return pickled_dataset

    def _build_pickled_dataset(self, pickled_dataset, options={}):
        """ Build a dataset where images are loaded from a pickled file """
        nb_images, nb_train, nb_valid, nb_test = self._get_dataset_length(pickled_dataset)
        train_dataset = {'data': np.zeros((nb_train, 3, MAX_HEIGHT, MAX_WIDTH), dtype=theano.config.floatX), 'captions': []}
        valid_dataset = {'data': np.zeros((nb_valid, 3, MAX_HEIGHT, MAX_WIDTH), dtype=theano.config.floatX), 'captions': []}
        test_dataset = {'data': np.zeros((nb_test, 3, MAX_HEIGHT, MAX_WIDTH), dtype=theano.config.floatX), 'captions': []}

        for i in range(nb_images):
            if i % 100 == 99:
                print('..... Loaded %d of %d images from pickled dataset' % (i+1, nb_images))
            if i < nb_train:
                train_dataset['data'][i, :, :, :] = pickled_dataset['data'][i, :, :, :]
                train_dataset['captions'].append(pickled_dataset['captions'][i])
            elif i < nb_train + nb_valid:
                valid_dataset['data'][i - nb_train, :, :, :] = pickled_dataset['data'][i, :, :, :]
                valid_dataset['captions'].append(pickled_dataset['captions'][i])
            else:
                test_dataset['data'][i - nb_train - nb_valid, :, :, :] = pickled_dataset['data'][i, :, :, :]
                test_dataset['captions'].append(pickled_dataset['captions'][i])
        print('... Done loading images from pickled dataset')

        # Shuffling
        train_permutation = np.random.permutation(np.arange(train_dataset['data'].shape[0]))
        valid_permutation = np.random.permutation(np.arange(valid_dataset['data'].shape[0]))
        test_permutation = np.random.permutation(np.arange(test_dataset['data'].shape[0]))

        train_dataset['data'] = train_dataset['data'][train_permutation]
        train_dataset['captions'] = [train_dataset['captions'][i] for i in train_permutation]
        valid_dataset['data'] = valid_dataset['data'][valid_permutation]
        valid_dataset['captions'] = [valid_dataset['captions'][i] for i in valid_permutation]
        test_dataset['data'] = test_dataset['data'][test_permutation]
        test_dataset['captions'] = [test_dataset['captions'][i] for i in test_permutation]

        return pickled_dataset, train_dataset, valid_dataset, test_dataset

    def _build_disk_dataset(self, coco_dataset, options={}):
        """ Build a dataset where images are stored on disk and loaded from disk at every batch - Supports dataset augmentation  """
        nb_images, nb_train, nb_valid, nb_test = self._get_dataset_length(coco_dataset)
        train_dataset = {'data': [], 'captions': []}
        valid_dataset = {'data': [], 'captions': []}
        test_dataset = {'data': [], 'captions': []}
        dataset = {}

        for i in range(nb_images):
            if i % 100 == 99:
                print('..... Loaded %d of %d images summary from disk' % (i+1, nb_images))
            curr_image = coco_dataset[i]
            curr_image['full_path'] = self._get_image_full_path(i, options, curr_image['file_name'])
            curr_captions = self._get_captions(i, options, curr_image['id'])

            if i < nb_train:
                train_dataset['data'].append(curr_image)
                train_dataset['captions'].append(curr_captions)
            elif i < nb_train + nb_valid:
                valid_dataset['data'].append(curr_image)
                valid_dataset['captions'].append(curr_captions)
            else:
                test_dataset['data'].append(curr_image)
                test_dataset['captions'].append(curr_captions)
        print('... Done loading images summary from disk')

        # Shuffling
        train_permutation = np.random.permutation(np.arange(len(train_dataset['data'])))
        valid_permutation = np.random.permutation(np.arange(len(valid_dataset['data'])))
        test_permutation = np.random.permutation(np.arange(len(test_dataset['data'])))

        train_dataset['data'] = [train_dataset['data'][i] for i in train_permutation]
        train_dataset['captions'] = [train_dataset['captions'][i] for i in train_permutation]
        valid_dataset['data'] = [valid_dataset['data'][i] for i in valid_permutation]
        valid_dataset['captions'] = [valid_dataset['captions'][i] for i in valid_permutation]
        test_dataset['data'] = [test_dataset['data'][i] for i in test_permutation]
        test_dataset['captions'] = [test_dataset['captions'][i] for i in test_permutation]

        return dataset, train_dataset, valid_dataset, test_dataset

    def _generate_samples_worker(self, nb_samples, dataset_name, skip_images, results, index):
        """ Returns augmented samples (images with captions) """
        t_nb_samples = nb_samples[index]
        results[index] = (np.zeros(shape=(t_nb_samples, 3, MAX_HEIGHT, MAX_WIDTH)), [], [], [])     # Img, Real, Matching, Fake Caption

        for ix in range(t_nb_samples):
            curr_ix = py_rng.randint(0, self.nb_examples[dataset_name] - 1)
            fake_ix = py_rng.randint(0, self.nb_examples[dataset_name] - 1)
            if curr_ix == fake_ix:
                fake_ix = (fake_ix + 1) % self.nb_examples[dataset_name]

            if not skip_images:
                keep_ratio = True if py_rng.random() >= 0.5 else False
                offset_h, offset_v = py_rng.randint(0, int(0.25 * MAX_HEIGHT)), py_rng.randint(0, int(0.25 * MAX_HEIGHT))
                flip_img = True if py_rng.random() >= 0.5 else False
                gray_scale = True if py_rng.random() >= 0.9 else False
                rotate_angle = py_rng.random() * 30. - 15.  # -15 to +15

                img = Image(self.dataset[dataset_name]['data'][curr_ix])
                img.resize(int(1.25 * MAX_WIDTH), int(1.25 * MAX_HEIGHT), keep_ratio=keep_ratio)
                if gray_scale:
                    img.to_gray()
                if flip_img:
                    img.flip()
                img.rotate(rotate_angle)
                img.crop(offset_h, offset_v, MAX_WIDTH, MAX_HEIGHT)
                results[index][0][ix, :, :, :] = img.data

            py_rng.shuffle(self.dataset[dataset_name]['captions'][curr_ix])
            results[index][1].append(self.dataset[dataset_name]['captions'][curr_ix][0].lower())
            if len(self.dataset[dataset_name]['captions'][curr_ix]) >= 2:
                results[index][2].append(self.dataset[dataset_name]['captions'][curr_ix][1].lower())
            else:
                results[index][2].append(self.dataset[dataset_name]['captions'][curr_ix][0].lower())
            results[index][3].append(py_rng.choice(self.dataset[dataset_name]['captions'][fake_ix]).lower())

    def _fill_buffer_worker(self, batch_size, dataset_name, img_scale):
        """ Fills the augmented sample buffer in the background """
        end_of_epoch = False
        while not end_of_epoch:
            while self.buffer[dataset_name].qsize() >= 3:          # To avoid loading too many images in memory
                time.sleep(1)
            next_batch = self.get_next_batch(batch_size, dataset_name, img_scale=img_scale, print_status=False)
            end_of_epoch = next_batch[4]
            if next_batch[0] is not None:
                self.buffer[dataset_name].put(next_batch)
