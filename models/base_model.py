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

import pickle as cPickle
import datetime
import lib
import logging
import math
import numpy as np
import os
import random
import settings
import string
import subprocess
import theano
import warnings
from settings import MAX_HEIGHT, MAX_WIDTH, SAVED_DIR, LOAD_DATASET_IN_MEMORY

class BaseModel(object):
    """ This is the base model class """

    def __init__(self):
        self.hparams = self._get_default_hparams()
        self.experiment_name = datetime.datetime.today().strftime('%Y-%m-%d_')
        self.experiment_name += ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(10))
        self.tparams = {}

    def save(self, desc=''):
        """ Save model to disk """
        current_date = datetime.datetime.today().strftime('%Y-%m-%d_%H-%M-%S')
        if not os.path.exists(os.path.join(SAVED_DIR, self.experiment_name)):
            os.makedirs(os.path.join(SAVED_DIR, self.experiment_name))
        file_name = os.path.join(SAVED_DIR, self.experiment_name, 'model.pkl')
        internal_dict = {}
        internal_dict['hparams'] = self.hparams
        internal_dict['tparams'] = { k: self.tparams[k].get_value() for k in self.tparams.keys() }
        with open(file_name, 'wb') as f:
            cPickle.dump(internal_dict, f, 2)
        #if len(desc) > 0:
        #    file_name = os.path.join(SAVED_DIR, self.experiment_name, 'model-%s-%s.pkl' % (desc, current_date))
        #else:
        #    file_name = os.path.join(SAVED_DIR, self.experiment_name, 'model-%s.pkl' % (current_date))
        #with open(file_name, 'wb') as f:
        #    cPickle.dump(internal_dict, f, 2)
        file_name = os.path.join(SAVED_DIR, self.experiment_name, 'hparams.txt')
        #git_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('utf-8')[0:-1]
        model_name = self.__class__.__name__
        with open(file_name, 'w') as f:
        #    f.write("%s:\t\t\t%s\n" % ('GIT Commit:', git_commit))
        #    f.write("%s:\t\t\t%s\n" % ('Model name:', model_name))
            for hparam in sorted(self.hparams.keys()):
                f.write("%s:\t\t\t%s\n" % (hparam, self.hparams[hparam]))
            for s in sorted(dir(settings)):
                if not s.startswith('_'):
                    f.write("SETTINGS-%s:\t\t\t%s\n" % (s, getattr(settings, s)))
        print('Model saved to %s' % (self.experiment_name))

    def load(self, filename='model.pkl', build_model=False):
        """ Load model from disk """
        print('... Loading model from disk')
        if os.path.exists(filename):
            model_path = filename
        else:
            model_path = os.path.join(SAVED_DIR, filename)
        if not os.path.exists(model_path):
            warnings.warn('Unable to find model file %s' % (model_path))
        with open(model_path, 'rb') as f:
            loaded_model = cPickle.load(f)
        self.hparams.update(loaded_model['hparams'])
        if build_model:
            self.build()
        for k in loaded_model['tparams'].keys():
            if k in self.tparams:
                self.tparams[k].set_value(loaded_model['tparams'][k])
            else:
                print('..... Model parameter %s is not in the Theano graph anymore. Skipping.' % (k))
        print('... Model %s successfully loaded from disk' % (model_path))

    def randomize(self):
        """ Use random hyperparameters (for random search) """
        self.hparams.update(self._get_random_hparams())

    def build(self):
        """ Build the model """
        raise ('Not implemented in BaseModel.')

    def run(self):
        """ Run the model"""
        raise('Not implemented')

    def evaluate(self, callback=None):
        """ Evaluate a model with random parameters """
        self.randomize()
        eval_keys = [k for k in self.hparams if 'eval_' in k]
        for k in eval_keys:
            del self.hparams[k]
        process = self.run if callback is None else getattr(self, callback)
        try:
            self.build()
            start = datetime.datetime.now()
            train_cost, best_valid_loss = process()
            end = datetime.datetime.now()
            diff = end - start
            print('... Total time taken to run: %d seconds' % (diff.seconds))
            self.hparams['eval_train_cost'] = train_cost
            self.hparams['eval_best_valid_loss'] = best_valid_loss
            self.hparams['eval_time_taken'] = diff.seconds
            self.hparams['eval_score'] = best_valid_loss * math.sqrt(diff.seconds)
            return
        except Exception as ex:
            logging.exception('Unable to complete evaluate().')
            self.hparams['eval_train_cost'] = np.inf
            self.hparams['eval_best_valid_loss'] = np.inf
            self.hparams['eval_time_taken'] = 0
            self.hparams['eval_score'] = 999999999

    # -----------------------
    # --- Private Methods ---
    # -----------------------
    def _get_default_hparams(self):
        """ Returns default hyperparameters """
        return {}

    def _get_random_hparams(self):
        """ Returns random hyperparameters """
        raise('Not implemented in BaseModel.')

    def _get_src_images_from_disk(self, dataset, batch_ix, batch_size):
        """ Load images from disk for a given batch id

        Args: dataset - The trainining, validation or test set
              batch_ix - The current batch id
              batch_size - The current batch size

        Returns a np array of size (batch_size, 3, h, w) with images
        """
        if LOAD_DATASET_IN_MEMORY:
            raise('Dataset already loaded in memory')
        src_images = np.zeros((batch_size, 3, MAX_HEIGHT, MAX_WIDTH), dtype=theano.config.floatX)
        for i, ix in enumerate(range(batch_ix * batch_size, (batch_ix + 1) * batch_size)):
            src_images[i, :, :, :] = lib.Image(dataset['data'][ix]).data
        return src_images
