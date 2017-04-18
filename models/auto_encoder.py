import collections
import lib
from lib.inits import constant, he
from lib.layers import relu, conv, max_pool, sigmoid, unpool
from lib.rng import py_rng
from lib.updates import Adam
from lib.utils import castX, shared0s
import math
import numpy as np
import theano
import theano.tensor as T
from . import BaseModel
from settings import MAX_HEIGHT, MAX_WIDTH, SAMPLES_TO_GENERATE, SAVE_MODEL_TO_DISK

class AutoEncoder(BaseModel):
    """ This is the main object class """

    def build(self):
        print('... Loading dataset')
        self.loader = lib.Loader(self.hparams)
        self.loader.load_dataset()
        print('... Done loading dataset')

        print('----- Number of examples in training set: %d' % (self.loader.nb_examples['train']))
        print('----- Number of examples in validation set: %d' % (self.loader.nb_examples['valid']))
        print('----- Number of examples in test set: %d' % (self.loader.nb_examples['test']))

        print('... Current Hyper-Parameters:')
        for k in sorted(self.hparams.keys()):
            print('..... %s: %s' % (k, self.hparams[k]))

        print('... Building model')
        # Getting hyperparameters
        batch_size = self.hparams.get('batch_size')
        nb_iterations = self.hparams.get('nb_iterations')
        learning_rate = self.hparams.get('learning_rate')
        input_maps = 3                      # 3 channels
        conv1_maps = self.hparams.get('conv1_maps')
        conv2_maps = self.hparams.get('conv2_maps')
        ae_hidden = self.hparams.get('ae_hidden')
        ae_visible = conv2_maps * MAX_HEIGHT * MAX_WIDTH // 16

        # Theano symbolic variables
        index = T.iscalar('index')
        src_images = T.tensor4('src_images', dtype=theano.config.floatX)            # Target (nb_batch, 3, h, w)
        current_batch_size = src_images.shape[0]
        o = T.alloc(np.asarray(1., dtype=theano.config.floatX), current_batch_size, 3, MAX_HEIGHT, MAX_WIDTH)
        self.gpu_dataset = shared0s(shape=(10 * batch_size, 3, MAX_HEIGHT, MAX_WIDTH))

        # Mask (1. for inner rectangle, 0 otherwise)
        mask_perc_h = self.hparams.get('mask_perc_h', 0.5)
        mask_perc_v = self.hparams.get('mask_perc_v', 0.5)
        mask_w = int(round(MAX_WIDTH * mask_perc_h))
        mask_h = int(round(MAX_HEIGHT * mask_perc_v))
        offset_h = int(round((MAX_WIDTH - mask_w) // 2))
        offset_v = int(round((MAX_HEIGHT  - mask_h) // 2))
        mask = T.zeros((current_batch_size, 3, MAX_HEIGHT, MAX_WIDTH), dtype=theano.config.floatX)
        mask = T.set_subtensor(mask[:, :, offset_v : offset_v + mask_h, offset_h : offset_h + mask_w], 1.)

        # Theano parameters
        self.tparams = collections.OrderedDict()
        self.tparams['W_conv1'] = he(shape=(conv1_maps, input_maps, 3, 3), fan_in=(input_maps*3.*3.), name='W_conv1')
        self.tparams['b_conv1'] = constant(shape=(conv1_maps,), c=0., name='b_conv1')
        self.tparams['W_conv2'] = he(shape=(conv2_maps, conv1_maps, 3, 3), fan_in=(conv1_maps*3.*3.), name='W_conv2')
        self.tparams['b_conv2'] = constant(shape=(conv2_maps,), c=0., name='b_conv2')
        self.tparams['W_encoder'] = he(shape=(ae_visible, ae_hidden), fan_in=(ae_hidden), name='W_encoder')
        self.tparams['b_encoder_hid'] = he(shape=(ae_hidden,), fan_in=(ae_visible), name='b_encoder_hid')
        self.tparams['b_encoder_vis'] = he(shape=(ae_visible,), fan_in=(ae_hidden), name='b_encoder_vis')
        self.tparams['W_conv3'] = he(shape=(conv1_maps, conv2_maps, 3, 3), fan_in=(conv2_maps*3.*3.), name='W_conv3')
        self.tparams['b_conv3'] = constant(shape=(conv1_maps,), c=0., name='b_conv3')
        self.tparams['W_conv4'] = he(shape=(input_maps, conv1_maps, 3, 3), fan_in=(conv1_maps*3.*3.), name='W_conv4')
        self.tparams['b_conv4'] = constant(shape=(input_maps,), c=0., name='b_conv4')
        tp = self.tparams

        # Auto-encoder
        x_t = castX(((T.zeros_like(mask) * mask) + (1. - mask) * src_images))                   # Resetting - output - size (b, 3, h, w)
        h1 = max_pool(relu(conv(X=x_t, w=tp['W_conv1'], b=tp['b_conv1'])), ws=(2, 2))           # output (b, conv1_maps, h/2, w/2)
        h2 = max_pool(relu(conv(X=h1, w=tp['W_conv2'], b=tp['b_conv2'])), ws=(2, 2))            # output (b, conv2_maps, h/4, w/4)
        h3 = sigmoid(T.dot(h2.flatten(2), tp['W_encoder']) + tp['b_encoder_hid'])
        h4 = sigmoid(T.dot(h3, tp['W_encoder'].T) + tp['b_encoder_vis']).reshape(h2.shape)      # output (b, conv2_maps, h/4, w/4)
        h5 = relu(conv(X=unpool(h4, us=(2, 2)), w=tp['W_conv3'], b=tp['b_conv3']))              # output (b, conv1_maps, h/2, w/2)
        h6 = relu(conv(X=unpool(h5, us=(2, 2)), w=tp['W_conv4'], b=tp['b_conv4']))              # output (b, input_maps, h, w)

        out_images = castX(T.clip(((h6 * mask) + (1. - mask) * src_images), 0., 1.))
        cost = 1000. * T.mean( (out_images - src_images)**2 )
        updates = Adam(lr=learning_rate)(list(self.tparams.values()), cost)
        print('... Done building model')

        print('... Compiling model')
        self.train_fn = theano.function(
            inputs=[index],
            outputs=[cost, out_images],
            updates=updates,
            givens={
                src_images: self.gpu_dataset[index * batch_size: (index + 1) * batch_size]
            },
            on_unused_input='warn')

        self.validate_fn = theano.function(
            inputs=[index],
            outputs=cost,
            givens={
                src_images: self.gpu_dataset[index * batch_size: (index + 1) * batch_size]
            },
            on_unused_input='warn')

        self.test_fn = theano.function(
            inputs=[index],
            outputs=out_images,
            givens={
                src_images: self.gpu_dataset[index * batch_size: (index + 1) * batch_size]
            },
            on_unused_input='warn')
        print('... Done compiling model')

    def run(self):
        """ Run the model """
        print('... Running algorithm')

        # Calculating batch size
        batch_size = self.hparams.get('batch_size')
        nb_batch_train = int(math.ceil(self.loader.nb_examples['train'] / float(batch_size)))
        nb_batch_valid = int(math.ceil(self.loader.nb_examples['valid'] / float(batch_size)))
        nb_batch_test = int(math.ceil(max(0, min(SAMPLES_TO_GENERATE, self.loader.nb_examples['test'])) / float(batch_size)))
        nb_epochs = self.hparams.get('nb_epochs')

        # Early stopping parameters
        patience = nb_batch_train * self.hparams.get('patience')    # Minimum number of batch to run before looking to stop early
        patience_increase = self.hparams.get('patience_increase')   # If best model is found, multiply patience by this factor
        improvement_threshold = self.hparams.get('improvement_threshold')
        validation_frequency = 200                                  # Validate every epoch or every 2000 mini-batches
        best_validation_loss = np.inf
        train_cost = np.inf
        initial_train_cost = np.inf
        abort_multiple = 3.

        # Function shorthands
        save_image = lib.Image().save
        display_image = lib.Image().display

        # Looping
        iter = -1
        for epoch in range(nb_epochs):
            self.loader.start_new_epoch()
            end_of_epoch = False
            batch_ix = -1
            master_batch_ix = -1

            # Running a full epoch
            while not end_of_epoch:

                # Retrieving a master batch of 10 mini-batches and storing them on the GPU
                master_batch_ix += 1
                imgs, real_captions, matching_captions, fake_captions, end_of_epoch = \
                    self.loader.get_next_batch(10 * batch_size, 'train', print_status=True)
                self.gpu_dataset.set_value(imgs)
                nb_train_iter = int(math.ceil(imgs.shape[0] / float(batch_size)))

                # Processing these 10 mini-batches
                for ix in range(nb_train_iter):
                    iter += 1
                    batch_ix += 1
                    train_cost, out_images = self.train_fn(ix)

                    if iter == 0:
                        initial_train_cost = train_cost
                    elif train_cost > abort_multiple * initial_train_cost:  # Unstable learning rate, aborting
                        print('... Unstable configuration (High learning rate). Aborting.')
                        return train_cost, best_validation_loss
                    print('Batch %04d/%04d - %04d/%04d - Epoch %04d -- Training Cost: %.4f' % (batch_ix + 1, nb_batch_train, iter + 1, patience, epoch, train_cost))

                # Checking if we need to validate
                if (master_batch_ix + 1) % validation_frequency == 0 or end_of_epoch:
                    valid_cost = []
                    end_of_validation = False
                    valid_iter = -1
                    if nb_batch_valid > 100:
                        print('... Running validation. Please be patient.')

                    # Retrieving 10 mini-batches of validation and calculating their cost
                    while not end_of_validation:
                        imgs, real_captions, matching_captions, fake_captions, end_of_validation = \
                            self.loader.get_next_batch(10 * batch_size, 'valid', print_status=False)
                        self.gpu_dataset.set_value(imgs)
                        nb_valid_iter = int(math.ceil(imgs.shape[0] / float(batch_size)))
                        for valid_ix in range(nb_valid_iter):
                            valid_iter += 1
                            valid_cost.append(self.validate_fn(valid_ix))

                            if valid_ix % 10 == 9:
                                print('... Validation Batch %04d/%04d - %04d/%04d - Epoch %04d -- Current Validation Cost: %.4f' %
                                      (valid_iter + 1, nb_batch_valid, iter, patience, epoch, np.mean(valid_cost)))

                    if nb_batch_valid > 0:
                        valid_cost = np.mean(valid_cost)
                        print('Batch %04d/%04d - %04d/%04d - Epoch %04d -- Training / Validation Cost: %.4f - %.4f' %
                              (batch_ix + 1, nb_batch_train, iter + 1, patience, epoch, train_cost, valid_cost))
                    else:
                        print('... Not enough validation data to run a full batch')

                    if nb_batch_valid > 100:
                        print('... Done validation.')
                    if valid_cost < best_validation_loss * improvement_threshold:
                        patience = max(patience, iter * patience_increase)
                        best_validation_loss = valid_cost
                        print('***** Better model found - Validation loss %.4f' % (best_validation_loss))
                        if SAVE_MODEL_TO_DISK:
                            self.save()

            # Generating test samples
            if nb_batch_test > 0:
                print('... Generating test samples.')
                end_of_test = False
                test_iter = -1
                sample_iter = -1
                while not end_of_test:
                    imgs, real_captions, matching_captions, fake_captions, end_of_test = \
                        self.loader.get_next_batch(10 * batch_size, 'test', print_status=False)
                    self.gpu_dataset.set_value(imgs)
                    nb_test_iter = int(math.ceil(imgs.shape[0] / float(batch_size)))
                    for test_ix in range(nb_test_iter):
                        test_iter += 1
                        out_images = self.test_fn(test_ix)
                        for i in range(out_images.shape[0]):
                            sample_iter += 1
                            save_image('test-epoch-%04d' % (epoch), '-id-%04d' % (sample_iter + 1), out_images[i])
                print('... Done generating test samples.')

            if iter >= patience:
                return train_cost, best_validation_loss
        return train_cost, best_validation_loss


    # -----------------------
    # --- Private Methods ---
    # -----------------------
    def _get_default_hparams(self):
        """ Returns default hyperparameters"""
        return {
            'ae_hidden':            250,
            'batch_size':           32,
            'conv1_maps':           16,
            'conv2_maps':           32,
            'improvement_threshold': 0.995,
            'learning_rate':        0.0009,
            'mask_perc_h':          0.5,
            'mask_perc_v':          0.5,
            'nb_epochs':            500,
            'nb_iterations':        2,
            'patience':             5,
            'patience_increase':    2.,
            'perc_train':           0.85,
            'perc_valid':           0.10,
            'perc_test':            0.05
        }

    def _get_random_hparams(self):
        hparams = {}
        hparams['ae_hidden'] = 20 * py_rng.randint(5, 25)
        hparams['batch_size'] = 2 ** py_rng.randint(2, 7)
        hparams['conv1_maps'] = 2 ** py_rng.randint(3, 6)
        hparams['conv2_maps'] = 2 ** py_rng.randint(3, 6)
        hparams['improvement_threshold'] = py_rng.choice([0.99, 0.999, 0.995, 0.9999, 1.])
        hparams['learning_rate'] = py_rng.choice([1., 3., 5., 7.]) * 10 ** py_rng.randint(-5, -3)
        hparams['nb_epochs'] = 1
        hparams['nb_iterations'] = py_rng.randint(1, 4)
        return hparams
