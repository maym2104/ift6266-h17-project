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

import collections
import lib
from lib.inits import constant, he, normal, orthogonal
from lib.layers import elu, relu, conv, conv_1d, sigmoid, softmax, tanh, batchnorm, concat, max_pool_1d, dropout, bilinear_upsample, CategoricalCrossEntropy, MeanSquaredError
from lib.rng import t_rng, t_rng_2
from lib.updates import Adam
from lib.utils import castX, shared0s, sharedX, floatX
import math
import numpy as np
import theano
import theano.tensor as T
from theano.ifelse import ifelse
from . import BaseModel
from settings import MAX_HEIGHT, MAX_WIDTH, SAMPLES_TO_GENERATE, SAVE_MODEL_TO_DISK
upsample = bilinear_upsample

class GenerativeAdversarialNetworkv2(BaseModel):
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
        # General
        batch_size = self.hparams.get('batch_size')
        nb_batch_store_gpu = self.hparams.get('nb_batch_store_gpu')
        nb_batch_train = int(math.ceil(self.loader.nb_examples['train'] / float(batch_size)))
        lb_g_cost_disc = self.hparams.get('lb_g_cost_disc')
        lb_g_feature = self.hparams.get('lb_g_feature')
        lb_d_cost_data_real = self.hparams.get('lb_d_cost_data_real')
        lb_d_cost_data_fake = self.hparams.get('lb_d_cost_data_fake')
        lb_d_cost_gen_real = self.hparams.get('lb_d_cost_gen_real')
        adam_b1 = self.hparams.get('adam_b1')
        emb_adam_b1 = self.hparams.get('emb_adam_b1')

        # GAN
        z_dim = self.hparams.get('z_dim')                                           # Noise dimensions
        mb_nb_kernels = self.hparams.get('mb_nb_kernels')                           # Number of kernels for mini-batch discrimination
        mb_kernel_dim = self.hparams.get('mb_kernel_dim')                           # Dimension of each mini-batch kernel

        # Embedding
        emb_cnn_dim = self.hparams.get('emb_cnn_dim')
        emb_dim = self.hparams.get('emb_dim')
        emb_doc_length = self.hparams.get('emb_doc_length')

        # Fixed
        input_maps = 3  # 3 channels
        emb_alphabet = 'abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:\'"/\\|_@#$%^&*~`+-=<>()[]{} '
        emb_char_to_ix = {ch: ix for ix, ch in enumerate(emb_alphabet)}
        emb_alphabet_size = len(emb_alphabet)

        # Theano symbolic variables
        index = T.iscalar('index')
        z = T.matrix('z', dtype=theano.config.floatX)                               # Noise  (nb_batch, z_dim)
        src_images = T.tensor4('src_images', dtype=theano.config.floatX)            # Target (nb_batch, 3, h, w)
        one_hot_1 = T.tensor3('one_hot_1', dtype=theano.config.floatX)              # One hot vector (nb_batch, doc_length, alphabet_size)
        one_hot_2 = T.tensor3('one_hot_2', dtype=theano.config.floatX)              # One hot vector (nb_batch, doc_length, alphabet_size)
        current_batch_size = src_images.shape[0]

        # Mask (1. for inner rectangle, 0 otherwise)
        mask = T.zeros(shape=(current_batch_size, 1, MAX_HEIGHT, MAX_WIDTH), dtype=theano.config.floatX)
        mask = T.set_subtensor(mask[:, :, 16:48, 16:48], 1.)
        mask_color = castX(T.cast(t_rng.uniform(size=(current_batch_size,), low=0., high=2.), 'int16').dimshuffle(0, 'x', 'x', 'x') * mask)

        # Creating space for storing dataset on GPU
        self.gpu_dataset = shared0s(shape=(batch_size * nb_batch_store_gpu, input_maps, MAX_HEIGHT, MAX_WIDTH), target='dev1')
        self.emb_dataset = shared0s(shape=(batch_size * nb_batch_store_gpu, 2, emb_doc_length, emb_alphabet_size), target='dev0')

        # Theano parameters
        self.tparams = collections.OrderedDict()
        self.tparams['hyper_emb_learning_rate'] = sharedX(self.hparams.get('emb_learning_rate'), name='emb_learning_rate', target='dev0')
        self.tparams['hyper_learning_rate'] = sharedX(self.hparams.get('learning_rate'), name='learning_rate', target='dev1')

        # Embedding
        self.tparams['emb_cnn_01_conv'] = he(shape=(384, emb_alphabet_size, 4), fan_in=(384 * 4), name='emb_cnn_01_conv', target='dev0')
        self.tparams['emb_cnn_02_conv'] = he(shape=(512, 384, 4), fan_in=(512 * 4), name='emb_cnn_02_conv', target='dev0')
        self.tparams['emb_cnn_03_conv'] = he(shape=(emb_cnn_dim, 512, 4), fan_in=(emb_cnn_dim * 4), name='emb_cnn_03_conv', target='dev0')
        self.tparams['emb_lstm_W'] = sharedX(np.concatenate([orthogonal(shape=(emb_cnn_dim, emb_cnn_dim), scale=1., shared=False),                # f, i, o ,g
                                                             orthogonal(shape=(emb_cnn_dim, emb_cnn_dim), scale=1., shared=False),
                                                             orthogonal(shape=(emb_cnn_dim, emb_cnn_dim), scale=1., shared=False),
                                                             orthogonal(shape=(emb_cnn_dim, emb_cnn_dim), scale=1., shared=False)], axis=1),
                                             name='emb_lstm_W',
                                             target='dev0')
        self.tparams['emb_lstm_U'] = sharedX(np.concatenate([orthogonal(shape=(emb_cnn_dim, emb_cnn_dim), scale=1., shared=False),                # f, i, o ,g
                                                             orthogonal(shape=(emb_cnn_dim, emb_cnn_dim), scale=1., shared=False),
                                                             orthogonal(shape=(emb_cnn_dim, emb_cnn_dim), scale=1., shared=False),
                                                             orthogonal(shape=(emb_cnn_dim, emb_cnn_dim), scale=1., shared=False)], axis=1),
                                             name='emb_lstm_U',
                                             target='dev0')
        self.tparams['emb_lstm_b'] = sharedX(np.concatenate([constant(shape=(emb_cnn_dim,), c=1., shared=False),                                  # f, i, o, g
                                                             constant(shape=(emb_cnn_dim,), c=0., shared=False),
                                                             constant(shape=(emb_cnn_dim,), c=0., shared=False),
                                                             constant(shape=(emb_cnn_dim,), c=0., shared=False)]),
                                             name='emb_lstm_b',
                                             target='dev0')
        self.tparams['emb_lstm_02_W'] = he(shape=(emb_cnn_dim, emb_dim), fan_in=(emb_cnn_dim), name='emb_lstm_02_W', target='dev0')
        self.tparams['emb_lstm_02_b'] = constant(shape=(emb_dim,), c=0., name='emb_lstm_02_b', target='dev0')


        # Helper functions
        def create_batch_norm_params(layer_inputs, name, target, tp):
            tp[name + '_g'] = normal(shape=(layer_inputs,), mean=1.0, std_dev=0.02, name=(name + '_g'), target=target)
            tp[name + '_b'] = constant(shape=(layer_inputs,), c=0., name=(name + '_b'), target=target)
            return tp

        def create_conv_layer_params(src_inputs, dest_inputs, kernel, name, target, tp, use_batch_norm=True):
            tp[name + '_conv'] = he(shape=(dest_inputs, src_inputs, kernel, kernel), fan_in=(dest_inputs * kernel * kernel), name=(name + '_conv'), target=target)
            if use_batch_norm:
                tp = create_batch_norm_params(dest_inputs, name, target, tp)
            return tp

        def create_deconv_layer_params(src_inputs, dest_inputs, kernel, name, target, tp, use_batch_norm=True):
            tp[name + '_conv'] = he(shape=(src_inputs, dest_inputs, kernel, kernel), fan_in=(src_inputs * kernel * kernel), name=(name + '_conv'), target=target)
            if use_batch_norm:
                tp = create_batch_norm_params(dest_inputs, name, target, tp)
            return tp

        def create_dense_layer_params(src_inputs, dest_inputs, name, target, tp, use_batch_norm=False, use_bias=False):
            tp[name + '_W'] = he(shape=(src_inputs, dest_inputs), fan_in=(src_inputs), name=(name + '_W'), target=target)
            if use_bias:
                tp[name + '_bi'] = constant(shape=(dest_inputs,), c=0., name=(name + '_bi'), target=target)
            if use_batch_norm:
                tp = create_batch_norm_params(dest_inputs, name, target, tp)
            return tp

        # Generator Parameters
        self.tparams = create_conv_layer_params(3, 64, 5, 'gen_01', 'dev1', self.tparams, use_batch_norm=True)
        self.tparams = create_conv_layer_params(64, 128, 5, 'gen_02', 'dev1', self.tparams, use_batch_norm=True)
        self.tparams = create_conv_layer_params(128, 256, 5, 'gen_03', 'dev1', self.tparams, use_batch_norm=True)
        self.tparams = create_conv_layer_params(256, 512, 5, 'gen_04', 'dev1', self.tparams, use_batch_norm=True)
        self.tparams = create_dense_layer_params(512*4*4, 4096, 'gen_05', 'dev1', self.tparams, use_batch_norm=True, use_bias=False)
        self.tparams = create_dense_layer_params(4096 + emb_dim + z_dim, 1024*4*4, 'gen_06', 'dev1', self.tparams, use_batch_norm=True, use_bias=False)
        self.tparams = create_conv_layer_params(1024, 512, 5, 'gen_07', 'dev1', self.tparams, use_batch_norm=True)
        self.tparams = create_conv_layer_params(512, 256, 5, 'gen_08', 'dev1', self.tparams, use_batch_norm=True)
        self.tparams = create_conv_layer_params(256, 128, 5, 'gen_09', 'dev1', self.tparams, use_batch_norm=True)
        self.tparams = create_conv_layer_params(128, 3, 5, 'gen_10', 'dev1', self.tparams, use_batch_norm=False)

        # Discriminator Parameters
        self.tparams = create_conv_layer_params(3, 64, 5, 'disc_01', 'dev0', self.tparams, use_batch_norm=False)
        self.tparams = create_conv_layer_params(64, 64, 5, 'disc_02', 'dev0', self.tparams, use_batch_norm=False)
        self.tparams = create_conv_layer_params(64, 64, 5, 'disc_03', 'dev0', self.tparams, use_batch_norm=False)
        self.tparams = create_conv_layer_params(64, 128, 5, 'disc_04', 'dev0', self.tparams, use_batch_norm=False)
        self.tparams = create_conv_layer_params(128, 128, 5, 'disc_05', 'dev0', self.tparams, use_batch_norm=False)
        self.tparams = create_conv_layer_params(128, 256, 5, 'disc_06', 'dev0', self.tparams, use_batch_norm=False)
        self.tparams = create_conv_layer_params(256, 256, 5, 'disc_07', 'dev0', self.tparams, use_batch_norm=False)
        self.tparams = create_conv_layer_params(256, 512, 5, 'disc_08', 'dev0', self.tparams, use_batch_norm=False)
        self.tparams = create_conv_layer_params(512, 512, 5, 'disc_09', 'dev0', self.tparams, use_batch_norm=False)
        self.tparams = create_conv_layer_params(512, 1024, 5, 'disc_10', 'dev0', self.tparams, use_batch_norm=False)
        self.tparams = create_dense_layer_params((1024 + emb_dim + mb_nb_kernels) * 4 * 4, 3, 'disc_12', 'dev0', self.tparams, use_batch_norm=False, use_bias=True)
        self.tparams['disc_10_mb_W'] = he(shape=(1024*4*4, mb_nb_kernels * mb_kernel_dim), fan_in=(1024*4*4), name='disc_10_mb_W', target='dev0')
        self.tparams['disc_10_mb_bi'] = constant(shape=(mb_nb_kernels,), c=0., name='disc_10_mb_b', target='dev0')

        # ---------------------
        # Embedding
        # ---------------------
        def _slice(x, n, dim):
            if x.ndim == 3:
                return x[:, :, n * dim:(n + 1) * dim]
            elif x.ndim == 2:
                return x[:, n * dim:(n + 1) * dim]
            raise NotImplementedError

        def one_hot_vector(list_of_captions):
            one_hot_vector = np.zeros(shape=(len(list_of_captions), emb_doc_length, emb_alphabet_size), dtype=theano.config.floatX)
            for ix, caption in enumerate(list_of_captions):
                caption = caption.lower()
                for j in range(emb_doc_length):
                    if j < len(caption) and caption[j] in emb_char_to_ix:
                        one_hot_vector[ix, j, emb_char_to_ix[caption[j]]] = 1
            return one_hot_vector                                                                   # (batch, doc_length, alpha_size)
        self.create_one_hot_vector = one_hot_vector

        def document_conv_net(one_hot, tp):                                                         # One-hot is (batch, doc_length, alpha_size)
            o_h0 = one_hot.dimshuffle(0, 2, 1)                                                      # (batch, 70, 201)
            o_h1 = relu(conv_1d(o_h0, tp['emb_cnn_01_conv'], subsample=(1,), border_mode='valid'))  # (batch, 384, 198) (kernel 384 x 70 x 4)
            o_h1 = max_pool_1d(o_h1, ws=3, stride=3)                                                # (batch, 384, 66)
            o_h2 = relu(conv_1d(o_h1, tp['emb_cnn_02_conv'], subsample=(1,), border_mode='valid'))  # (batch, 512, 63) (kernel 512 x 384 x 4)
            o_h2 = max_pool_1d(o_h2, ws=3, stride=3)                                                # (batch, 512, 21)
            o_h3 = relu(conv_1d(o_h2, tp['emb_cnn_03_conv'], subsample=(1,), border_mode='valid'))  # (batch, cnn_dim, 18) (kernel cnn_dim x 512 x 4)
            return o_h3

        def document_lstm_step(x_t, h_m1, c_m1, W, U, b):
            preact = T.dot(x_t, W) + T.dot(h_m1, U) + b                                             # (batch, 4*cnn_dim)
            f = sigmoid(_slice(preact, 0, emb_cnn_dim))
            i = sigmoid(_slice(preact, 1, emb_cnn_dim))
            o = sigmoid(_slice(preact, 2, emb_cnn_dim))
            g = tanh(_slice(preact, 3, emb_cnn_dim))
            c_t = f * c_m1 + i * g
            h_t = o * tanh(c_t)
            return h_t, c_t                                                                         # (batch, cnn_dim)

        def document_lstm(conv_hot, tp):                                                            # Input is (batch, cnn_dim, 18)
            l_h0 = conv_hot.dimshuffle(2, 0, 1)                                                     # Reshaping to (18, batch, cnn_dim)
            n_steps, n_batch = l_h0.shape[0], l_h0.shape[1]
            rval, updates = \
                theano.scan(
                    fn=document_lstm_step,
                    sequences=[l_h0],
                    outputs_info=[T.alloc(floatX(0.), n_batch, emb_cnn_dim),
                                  T.alloc(floatX(0.), n_batch, emb_cnn_dim)],
                    non_sequences=[tp['emb_lstm_W'], tp['emb_lstm_U'], tp['emb_lstm_b']],
                    n_steps=n_steps,
                    strict=True
                )                                                                                       # rval[0] is (nsteps, batch, cnn_dim)
            l_h1 = rval[0][-1]                                                                          # (batch, cnn_dim) - Embedding is the final h_t
            l_h2 = T.dot(l_h1, tp['emb_lstm_02_W']) + tp['emb_lstm_02_b']                               # (batch, emb_dim)
            return l_h2

        def joint_embedding_loss(emb_1, emb_2):
            score = T.dot(emb_1, emb_2.T)                                       # score[i, j] is dot product between emb_1[i, :] and emb_2[j, :]
            label_score = T.diag(score)                                         # label_score is score[i, i]
            loss = relu(score - label_score.dimshuffle(0, 'x') + 1.).sum()      # hinge loss - sum of (score[i, j] - label_score[i, i] + 1) if > 0
            best_score = T.argmax(score, axis=1)                                # position with smallest score in row i
            accuracy = 100. * T.eq(best_score, T.arange(emb_1.shape[0])).mean(dtype=theano.config.floatX)  # Number of examples where the best score is the label score
            return loss, accuracy


        # ---------------------
        # Generator
        # ---------------------
        def generator(emb, tp):
            Z = t_rng.normal((current_batch_size, z_dim),dtype=theano.config.floatX)
            g_h0  = castX((T.zeros_like(src_images) * mask) + (1. - mask) * src_images + mask_color)        # Image with mask cropped - b, 3, 64, 64
            g_h1  = conv(g_h0, tp['gen_01_conv'], subsample=(2, 2), border_mode=(2, 2))                     # b, 64, 32, 32
            g_h1  = elu(batchnorm(g_h1, g=tp['gen_01_g'], b=tp['gen_01_b']))                                # b, 64, 32, 32
            g_h2  = conv(g_h1, tp['gen_02_conv'], subsample=(2, 2), border_mode=(2, 2))                     # b, 128, 16, 16
            g_h2  = elu(batchnorm(g_h2, g=tp['gen_02_g'], b=tp['gen_02_b']))                                # b, 128, 16, 16
            g_h3  = conv(g_h2, tp['gen_03_conv'], subsample=(2, 2), border_mode=(2, 2))                     # b, 256, 8, 8
            g_h3  = elu(batchnorm(g_h3, g=tp['gen_03_g'], b=tp['gen_03_b']))                                # b, 256, 8, 8
            g_h4  = conv(g_h3, tp['gen_04_conv'], subsample=(2, 2), border_mode=(2, 2))                     # b, 512, 4, 4
            g_h4  = elu(batchnorm(g_h4, g=tp['gen_04_g'], b=tp['gen_04_b']))                                # b, 512, 4, 4
            g_h5  = T.dot(g_h4.flatten(2), tp['gen_05_W'])                                                  # b, 4096
            g_h5  = elu(batchnorm(g_h5, g=tp['gen_05_g'], b=tp['gen_05_b']))                                # b, 4096
            g_h5  = concat([Z, g_h5, emb], axis=1)                                                          # b, 4096 + emb_dim + z_dim
            g_h6  = T.dot(g_h5, tp['gen_06_W'])                                                             # b, 8192
            g_h6  = elu(batchnorm(g_h6, g=tp['gen_06_g'], b=tp['gen_06_b']))                                # b, 16384
            g_h6  = g_h6.reshape((current_batch_size, 1024, 4, 4))                                          # b, 1024, 4, 4
            g_h7  = conv(upsample(g_h6, 2), tp['gen_07_conv'], subsample=(1, 1), border_mode='half')        # b, 512, 8, 8
            g_h7  = elu(batchnorm(g_h7, g=tp['gen_07_g'], b=tp['gen_07_b']))                                # b, 512, 8, 8
            g_h8  = conv(upsample(g_h7, 2), tp['gen_08_conv'], subsample=(1, 1), border_mode='half')        # b, 256, 16, 16
            g_h8  = elu(batchnorm(g_h8, g=tp['gen_08_g'], b=tp['gen_08_b']))                                # b, 256, 16, 16
            g_h9  = conv(upsample(g_h8, 2), tp['gen_09_conv'], subsample=(1, 1), border_mode='half')        # b, 128, 32, 32
            g_h9  = elu(batchnorm(g_h9, g=tp['gen_09_g'], b=tp['gen_09_b']))                                # b, 128, 32, 32
            g_h10 = conv(upsample(g_h9, 2), tp['gen_10_conv'], subsample=(1, 1), border_mode='half')        # b, 3, 64, 64
            g_x = tanh(g_h10) * mask + (1. - mask) * src_images                                             # b, 3, 64, 64
            return g_x

        # ---------------------
        # Discriminator
        # ---------------------
        # Mini-batch discrimination
        # Adapted from https://github.com/openai/improved-gan/blob/master/mnist_svhn_cifar10/nn.py#L132
        def minibatch(d_h10, W, b):
            mb_h0 = T.dot(d_h10.flatten(2), W)                                                          # b, nb_kernels * kernel_dim
            mb_h0 = mb_h0.reshape((d_h10.shape[0], mb_nb_kernels, mb_kernel_dim))                       # b, nb_kernel, kernel_dim
            mb_h1 = mb_h0.dimshuffle(0, 1, 2, 'x') - mb_h0.dimshuffle('x', 1, 2, 0)
            mb_h1 = T.sum(abs(mb_h1), axis=2) + 1e6 * T.eye(d_h10.shape[0]).dimshuffle(0,'x',1)
            mb_h2 = T.sum(T.exp(-mb_h1), axis=2) + b                                                    # b, nb_kernel
            mb_h2 = mb_h2.dimshuffle(0, 1, 'x', 'x')
            mb_h2 = T.repeat(mb_h2, 4, axis=2)
            mb_h2 = T.repeat(mb_h2, 4, axis=3)
            return mb_h2

        def discriminator(g_x, emb, tp):
            d_emb = T.repeat(T.repeat(emb.dimshuffle(0, 1, 'x', 'x'), 4, axis=2), 4, axis=3)            # b, 1024, 4, 4
            d_h0 = dropout(g_x, p=0.2)
            d_h1 = elu(conv(d_h0, tp['disc_01_conv'], subsample=(1, 1), border_mode='half'))            # b, 64, 64, 64
            d_h2 = elu(conv(d_h1, tp['disc_02_conv'], subsample=(1, 1), border_mode='half'))            # b, 64, 64, 64
            d_h3 = elu(conv(d_h2, tp['disc_03_conv'], subsample=(2, 2), border_mode=(2, 2)))            # b, 64, 32, 32
            d_h3 = dropout(d_h3, p=0.5)                                                                 # b, 64, 32, 32
            d_h4 = elu(conv(d_h3, tp['disc_04_conv'], subsample=(1, 1), border_mode='half'))            # b, 128, 32, 32
            d_h5 = elu(conv(d_h4, tp['disc_05_conv'], subsample=(2, 2), border_mode=(2, 2)))            # b, 128, 16, 16
            d_h5 = dropout(d_h5, p=0.5)                                                                 # b, 128, 16, 16
            d_h6 = elu(conv(d_h5, tp['disc_06_conv'], subsample=(1, 1), border_mode='half'))            # b, 256, 16, 16
            d_h7 = elu(conv(d_h6, tp['disc_07_conv'], subsample=(2, 2), border_mode=(2, 2)))            # b, 256, 8, 8
            d_h7 = dropout(d_h7, p=0.5)                                                                 # b, 256, 8, 8
            d_h8 = elu(conv(d_h7, tp['disc_08_conv'], subsample=(1, 1), border_mode='half'))            # b, 512, 8, 8
            d_h9 = elu(conv(d_h8, tp['disc_09_conv'], subsample=(1, 1), border_mode='half'))            # b, 512, 8, 8
            d_h9 = dropout(d_h9, p=0.5)                                                                 # b, 512, 8, 8
            d_h10 = elu(conv(d_h9, tp['disc_10_conv'], subsample=(2, 2), border_mode=(2, 2)))           # b, 1024, 4, 4
            d_h10_mb = minibatch(d_h10, W=tp['disc_10_mb_W'], b=tp['disc_10_mb_bi'])                    # b, 1024, 4, 4
            d_h11 = concat([d_h10, d_h10_mb, d_emb], axis=1)                                            # b, 3072, 4, 4
            d_h11 = T.flatten(d_h11, 2)                                                                 # b, 32768
            d_h12 = softmax(T.dot(d_h11, tp['disc_12_W']) + tp['disc_12_bi'])                           # b, 3
            return d_h12, d_h10

        # ---------------------
        # Cost and updates
        # ---------------------
        flip_real_fake = T.gt(t_rng.uniform(size=(1,))[0], 0.95)
        flip_data_gen = T.gt(t_rng.uniform(size=(1,))[0], 0.95)

        # Source: https://github.com/fchollet/keras/issues/4329#issuecomment-269075723
        def unroll(updates, u_updates, depth):
            replace = {k: v for k, v in u_updates}
            updates_t = updates
            for i in range(depth):
                updates_t = [(k, theano.clone(v, replace)) for k, v in updates_t]
            return updates_t

        # Embedding cost and updates
        emb_1 = document_lstm(document_conv_net(one_hot_1, self.tparams), self.tparams)         # Real caption
        emb_2 = document_lstm(document_conv_net(one_hot_2, self.tparams), self.tparams)         # Matching caption (for embedding training), Fake caption (for disc. training)
        emb_loss, emb_accuracy = joint_embedding_loss(emb_1, emb_2)
        emb_params = [self.tparams[k] for k in self.tparams.keys() if k.startswith('emb_')]
        emb_updater = Adam(lr=self.tparams['hyper_emb_learning_rate'], b1=emb_adam_b1, clipnorm=10.)
        emb_updates = emb_updater(emb_params, emb_loss)

        # Different scenarios (real data, fake data)
        g_x = generator(emb_1, self.tparams)
        p_data_real, d_data_real_h10 = discriminator(ifelse(flip_data_gen, g_x, src_images), emb_1, self.tparams)
        p_data_fake, d_data_fake_h10 = discriminator(src_images, ifelse(flip_real_fake, emb_1, emb_2), self.tparams)
        p_gen_real, d_gen_real_h10   = discriminator(ifelse(flip_data_gen, src_images, g_x), emb_1, self.tparams)

        # Generator cost and updates
        choice_g = [0] * 9 + [1] + [2]
        g_cost_disc = CategoricalCrossEntropy(p_gen_real, t_rng_2.choice(size=(current_batch_size,), a=choice_g, dtype='int64'))
        g_feature_matching = MeanSquaredError(d_data_real_h10, d_gen_real_h10)                  # Feature matching (d_h3 of discriminator)
        g_total_cost  = lb_g_cost_disc * g_cost_disc
        g_total_cost += lb_g_feature * g_feature_matching
        accuracy_g_gen_real = 100. * T.mean(T.eq(T.argmax(p_gen_real, axis=1), 0.), dtype=theano.config.floatX)

        g_params = [self.tparams[k] for k in self.tparams.keys() if k.startswith('gen_')]
        g_updater = Adam(lr=self.tparams['hyper_learning_rate'], b1=adam_b1, clipnorm=10.)
        g_updates = g_updater(g_params, g_total_cost)

        # Discriminator cost and updates
        choice_data_real = [0] * 9 + [1] + [2]
        choice_data_fake = [1] * 9 + [0] + [2]
        choice_gen_real =  [2] * 9 + [0] + [1]
        d_cost_data_real = CategoricalCrossEntropy(p_data_real, t_rng_2.choice(size=(current_batch_size,), a=choice_data_real, dtype='int64'))          # Real image, real text (0.9 => One-Sided Label Smoothing)
        d_cost_data_fake = CategoricalCrossEntropy(p_data_fake, t_rng_2.choice(size=(current_batch_size,), a=choice_data_fake, dtype='int64'))          # Real image, fake text
        d_cost_gen_real  = CategoricalCrossEntropy(p_gen_real,  t_rng_2.choice(size=(current_batch_size,), a=choice_gen_real, dtype='int64'))           # Generator Image, real text
        d_total_cost  = lb_d_cost_data_real * d_cost_data_real
        d_total_cost += lb_d_cost_data_fake * d_cost_data_fake
        d_total_cost += lb_d_cost_gen_real * d_cost_gen_real
        accuracy_d_data_real = 100. * T.mean(T.eq(T.argmax(p_data_real, axis=1), 0.), dtype=theano.config.floatX)
        accuracy_d_data_fake = 100. * T.mean(T.eq(T.argmax(p_data_fake, axis=1), 1.), dtype=theano.config.floatX)
        accuracy_d_gen_real = 100. * T.mean(T.eq(T.argmax(p_gen_real, axis=1), 2.), dtype=theano.config.floatX)

        d_params = [self.tparams[k] for k in self.tparams.keys() if k.startswith('disc_')]
        d_updater = Adam(lr=self.tparams['hyper_learning_rate'], b1=adam_b1, clipnorm=10.)
        d_updates = d_updater(d_params, d_total_cost)

        # Unrolling updates (1 time step for generator)
        unrolled_g_updates = unroll(g_updates, d_updates, 1)
        unrolled_d_updates = unroll(d_updates, g_updates, 0)
        g_updates = unrolled_g_updates
        d_updates = unrolled_d_updates

        # Combined
        d_g_updates = g_updates + d_updates

        # Compilation variables
        self.cvars = {}
        cvars_list = ['index', 'src_images', 'one_hot_1', 'one_hot_2', 'batch_size', 'emb_loss', 'emb_accuracy',
                      'emb_updates', 'g_x', 'g_total_cost', 'g_updates', 'd_total_cost', 'd_updates', 'd_g_updates',
                      'g_cost_disc', 'g_feature_matching', 'd_cost_data_real', 'd_cost_data_fake', 'd_cost_gen_real',
                      'accuracy_g_gen_real', 'accuracy_d_data_real', 'accuracy_d_data_fake', 'accuracy_d_gen_real']
        for c in cvars_list:
            self.cvars[c] = locals()[c]

        print('... Done building model')


    def compile(self, fn_name):
        """ Compiling theano function """
        _ = self.cvars

        if fn_name == 'train_embedding' and not hasattr(self, 'train_embedding_fn'):
            print('..... Compiling train_embedding_fn()')
            self.train_embedding_fn = theano.function(
                inputs=[_['index']],
                outputs=[_['emb_loss'], _['emb_accuracy']],
                updates=_['emb_updates'],
                givens={
                    _['one_hot_1']: self.emb_dataset[_['index'] * _['batch_size']: (_['index'] + 1) * _['batch_size'], 0, :, :],
                    _['one_hot_2']: self.emb_dataset[_['index'] * _['batch_size']: (_['index'] + 1) * _['batch_size'], 1, :, :]
                },
                on_unused_input='ignore')
            print('..... Done compiling train_embedding_fn()')

        if fn_name == 'train_gen' and not hasattr(self, 'train_gen_fn'):
            print('..... Compiling train_gen_fn()')
            self.train_gen_fn = theano.function(
                inputs=[_['index']],
                outputs=[_['g_total_cost'], _['g_cost_disc'], _['g_feature_matching'], _['accuracy_g_gen_real']],
                updates=_['g_updates'],
                givens={
                    _['src_images']: self.gpu_dataset[_['index'] * _['batch_size']: (_['index'] + 1) * _['batch_size'], :, :, :],
                    _['one_hot_1']: self.emb_dataset[_['index'] * _['batch_size']: (_['index'] + 1) * _['batch_size'], 0, :, :],
                    _['one_hot_2']: self.emb_dataset[_['index'] * _['batch_size']: (_['index'] + 1) * _['batch_size'], 1, :, :]
                },
                on_unused_input='warn')
            print('..... Done compiling train_gen_fn()')

        if fn_name == 'train_disc' and not hasattr(self, 'train_disc_fn'):
            print('..... Compiling train_disc_fn()')
            self.train_disc_fn = theano.function(
                inputs=[_['index']],
                outputs=[_['d_total_cost'], _['d_cost_data_real'], _['d_cost_data_fake'], _['d_cost_gen_real'],
                         _['accuracy_d_data_real'], _['accuracy_d_data_fake'], _['accuracy_d_gen_real']],
                updates=_['d_updates'],
                givens={
                    _['src_images']: self.gpu_dataset[_['index'] * _['batch_size']: (_['index'] + 1) * _['batch_size'], :, :, :],
                    _['one_hot_1']: self.emb_dataset[_['index'] * _['batch_size']: (_['index'] + 1) * _['batch_size'], 0, :, :],
                    _['one_hot_2']: self.emb_dataset[_['index'] * _['batch_size']: (_['index'] + 1) * _['batch_size'], 1, :, :]
                },
                on_unused_input='warn')
            print('..... Done compiling train_disc_fn()')

        if fn_name == 'train_gen_and_disc' and not hasattr(self, 'train_gen_and_disc_fn'):
            print('..... Compiling train_gen_and_disc_fn()')
            self.train_gen_and_disc_fn = theano.function(
                inputs=[_['index']],
                outputs=[_['g_total_cost'], _['d_total_cost'], _['g_cost_disc'], _['g_feature_matching'], _['d_cost_data_real'],
                         _['d_cost_data_fake'], _['d_cost_gen_real'], _['accuracy_g_gen_real'], _['accuracy_d_data_real'],
                         _['accuracy_d_data_fake'], _['accuracy_d_gen_real']],
                updates=_['d_g_updates'],
                givens={
                    _['src_images']: self.gpu_dataset[_['index'] * _['batch_size']: (_['index'] + 1) * _['batch_size'], :, :, :],
                    _['one_hot_1']: self.emb_dataset[_['index'] * _['batch_size']: (_['index'] + 1) * _['batch_size'], 0, :, :],
                    _['one_hot_2']: self.emb_dataset[_['index'] * _['batch_size']: (_['index'] + 1) * _['batch_size'], 1, :, :]
                },
                on_unused_input='warn')
            print('..... Done compiling train_gen_and_disc_fn()')

        if fn_name == 'generate_samples' and not hasattr(self, 'generate_samples_fn'):
            print('..... Compiling generate_samples_fn()')
            self.generate_samples_fn = theano.function(
                inputs=[_['src_images'], _['one_hot_1']],
                outputs=_['g_x'],
                on_unused_input='warn')
            print('..... Done compiling generate_samples_fn()')


    def run_train_embedding(self):
        print('... Training Embedding')

        # Compiling
        self.compile('train_embedding')

        # Settings
        batch_size = self.hparams.get('batch_size')
        nb_batch_store_gpu = self.hparams.get('nb_batch_store_gpu')
        nb_batch_train = int(math.ceil(self.loader.nb_examples['train'] / float(batch_size)))
        nb_epochs = self.hparams.get('emb_nb_epochs')
        learning_rate_epoch = self.hparams.get('emb_learning_rate_epoch')
        learning_rate_adj = self.hparams.get('emb_learning_rate_adj')
        initial_learning_rate = self.hparams.get('emb_learning_rate')
        smooth_accuracy = 0.
        smooth_alpha = 0.99
        target_accuracy = 99.
        best_accuracy = -np.inf
        improvement_threshold = self.hparams.get('improvement_threshold')

        # Looping
        self.tparams['hyper_emb_learning_rate'].set_value(initial_learning_rate)
        iter = -1
        loss = 0.
        for epoch in range(nb_epochs):
            self.loader.start_new_epoch()
            end_of_epoch = False
            batch_ix = -1
            master_batch_ix = -1

            if (epoch + 1) % learning_rate_epoch == 0:
                learning_rate = self.tparams['hyper_emb_learning_rate'].get_value()
                self.tparams['hyper_emb_learning_rate'].set_value(np.float32(learning_rate * learning_rate_adj))

            # Running a full epoch
            while not end_of_epoch:

                # Retrieving a master batch of 10 mini-batches and storing them on the GPU
                master_batch_ix += 1
                imgs, real_captions, matching_captions, fake_captions, end_of_epoch = \
                    self.loader.get_next_batch(nb_batch_store_gpu * batch_size, 'train', skip_images=True, print_status=True)
                one_hot_real = self.create_one_hot_vector(real_captions)
                one_hot_matching = self.create_one_hot_vector(matching_captions)
                self.emb_dataset.set_value(np.concatenate([one_hot_real[:, None, :, :], one_hot_matching[:, None, :, :]], axis=1))
                nb_train_iter = int(math.ceil(imgs.shape[0] / float(batch_size)))

                # Processing these 10 mini-batches
                for ix in range(nb_train_iter):
                    iter += 1
                    batch_ix += 1
                    loss, accuracy = self.train_embedding_fn(ix)
                    smooth_accuracy = smooth_alpha * smooth_accuracy + (1. - smooth_alpha) * accuracy
                    print('Embedding Batch %04d/%04d - Epoch %04d -- Loss: %.4f - Accuracy: %.04f/%.04f - Best: %.4f' %
                          (batch_ix + 1, nb_batch_train, epoch + 1, loss, accuracy, smooth_accuracy, best_accuracy))
                if smooth_accuracy >= target_accuracy and iter >= 5:
                    return loss, smooth_accuracy
                if smooth_accuracy * improvement_threshold > best_accuracy and iter > 1:
                    best_accuracy = smooth_accuracy
                    print('***** Better embedding model found - Best Accuracy %.4f' % (smooth_accuracy))
                    if SAVE_MODEL_TO_DISK:
                        self.save('emb-%.4f' % (best_accuracy))

        return loss, smooth_accuracy

    def run_train_gan(self, border_only=False):
        """ Train the model """
        print('... Training Generative Adversarial Network')

        # Compiling
        train_fn = 'train_gen_border' if border_only else 'train_gen_and_disc'
        self.compile(train_fn)
        if max(0, min(SAMPLES_TO_GENERATE, self.loader.nb_examples['test'])) > 0:
            self.compile('generate_samples')

        # Calculating batch size
        batch_size = self.hparams.get('batch_size')
        nb_batch_store_gpu = self.hparams.get('nb_batch_store_gpu')
        nb_batch_train = int(math.ceil(self.loader.nb_examples['train'] / float(batch_size)))
        nb_batch_test = int(math.ceil(max(0, min(SAMPLES_TO_GENERATE, self.loader.nb_examples['test'])) / float(batch_size)))
        nb_epochs = self.hparams.get('nb_epochs')
        save_frequency = 50000 // (batch_size * nb_batch_store_gpu)

        learning_rate_epoch = self.hparams.get('learning_rate_epoch')
        learning_rate_adj = self.hparams.get('learning_rate_adj')
        initial_learning_rate = self.hparams.get('learning_rate')
        last_cost_g = []
        last_cost_d = []
        cost_g, cost_d = 0, 0

        # Function shorthands
        save_image = lib.Image().save
        display_image = lib.Image().display

        # Looping
        self.tparams['hyper_learning_rate'].set_value(initial_learning_rate)
        iter = -1
        if SAVE_MODEL_TO_DISK:
            self.save('gan-epoch-%04d.%04d' % (0, 0))

        for epoch in range(nb_epochs):
            self.loader.start_new_epoch()
            end_of_epoch = False
            batch_ix = -1
            master_batch_ix = -1
            starting_epoch = True

            if (epoch + 1) % learning_rate_epoch == 0:
                learning_rate = self.tparams['hyper_learning_rate'].get_value()
                self.tparams['hyper_learning_rate'].set_value(np.float32(learning_rate * learning_rate_adj))

            # Running a full epoch
            while not end_of_epoch:

                # Retrieving a master batch of 10 mini-batches and storing them on the GPU
                master_batch_ix += 1

                if starting_epoch:
                    imgs, real_captions, matching_captions, fake_captions, end_of_epoch = \
                        self.loader.get_next_batch(nb_batch_store_gpu * batch_size, 'train', skip_images=False, img_scale='tanh', print_status=True)
                    self.loader.fill_buffer(nb_batch_store_gpu * batch_size, img_scale='tanh')
                    starting_epoch = False
                else:
                    if self.loader.buffer['train'].qsize() > 0:
                        imgs, real_captions, matching_captions, fake_captions, end_of_epoch = self.loader.buffer['train'].get()
                    else:
                        print('..... Buffer is empty. Re-using current data for another batch.')
                        nb_train_iter = int(math.ceil(imgs.shape[0] / float(batch_size)))
                        master_batch_ix -= 1
                        iter -= nb_train_iter
                        batch_ix -= nb_train_iter

                # Storing data
                one_hot_real = self.create_one_hot_vector(real_captions)
                one_hot_fake = self.create_one_hot_vector(fake_captions)
                self.gpu_dataset.set_value(imgs)
                self.emb_dataset.set_value(np.concatenate([one_hot_real[:, None, :, :], one_hot_fake[:, None, :, :]], axis=1))
                nb_train_iter = int(math.ceil(imgs.shape[0] / float(batch_size)))

                # Processing these 10 mini-batches
                for ix in range(nb_train_iter):
                    iter += 1
                    batch_ix += 1
                    current_batch_size = imgs[ix * batch_size: (ix+ 1) * batch_size, :, :, :].shape[0]
                    cost_g, cost_d, g_cost_disc, g_feature_matching, d_cost_data_real, d_cost_data_fake, d_cost_gen_real, \
                        acc_g_gen_real, acc_d_data_real, acc_d_data_fake, acc_d_gen_real = self.train_gen_and_disc_fn(ix)
                    last_cost_g.append(cost_g)
                    last_cost_g = last_cost_g[-1000:]
                    last_cost_d.append(cost_d)
                    last_cost_d = last_cost_d[-1000:]
                    print('Batch %04d/%04d - Epoch %04d -- Loss G/D: %.4f/%.4f - Min/Max (1000): %.4f/%.4f - %.4f/%.4f ' %
                         (batch_ix + 1, nb_batch_train, epoch + 1, cost_g, cost_d, min(last_cost_g), max(last_cost_g), min(last_cost_d), max(last_cost_d)))
                    print('... GDisc) %.4f (%.2f%%) / GFeat) %.4f |  DReal) %.4f (%.2f%%) / DFake) %.4f (%.2f%%) / DGen) %.4f (%.2f%%)' %
                         (g_cost_disc, acc_g_gen_real, g_feature_matching, d_cost_data_real, acc_d_data_real, d_cost_data_fake, acc_d_data_fake, d_cost_gen_real, acc_d_gen_real))

                if (master_batch_ix + 1) % save_frequency == 0 or end_of_epoch:
                    if SAVE_MODEL_TO_DISK:
                        self.save('gan-epoch-%04d.%04d' % (epoch + 1, master_batch_ix + 1))

            # Generating test samples
            if nb_batch_test > 0:
                print('... Generating test samples.')
                end_of_test = False
                test_iter = -1
                sample_iter = -1
                test_batch_ix = 0
                while not end_of_test and test_batch_ix < nb_batch_test:
                    test_batch_ix += 1
                    imgs, real_captions, matching_captions, fake_captions, end_of_test = self.loader.buffer['test'].get()
                    one_hot_real = self.create_one_hot_vector(real_captions)
                    nb_test_iter = int(math.ceil(imgs.shape[0] / float(batch_size)))
                    for test_ix in range(nb_test_iter):
                        test_iter += 1
                        src_images = imgs[test_ix * batch_size: (test_ix + 1) * batch_size]
                        curr_emb = one_hot_real[test_ix * batch_size: (test_ix + 1) * batch_size]
                        out_images = self.generate_samples_fn(src_images, curr_emb)
                        for i in range(out_images.shape[0]):
                            sample_iter += 1
                            save_image('%s/samples-epoch-%04d' % (self.experiment_name, epoch + 1), '-id-%04d' % (sample_iter + 1), (out_images[i] + 1.) / 2.)
                print('... Done generating test samples.')

        return cost_g, cost_d

    def run(self):
        self.run_train_gan()

    # -----------------------
    # --- Private Methods ---
    # -----------------------
    def _get_default_hparams(self):
        """ Returns default hyperparameters"""
        return {
            'adam_b1':              0.5,
            'batch_size':           128,
            'emb_adam_b1':          0.5,
            'emb_cnn_dim':          512,
            'emb_dim':              1024,
            'emb_doc_length':       201,
            'emb_learning_rate':    0.0002,
            'emb_learning_rate_epoch':  10,
            'emb_learning_rate_adj':    0.75,
            'emb_nb_epochs':        500,
            'improvement_threshold': 0.995,
            'lb_g_cost_disc':       1.0,
            'lb_g_feature':         2.0,
            'lb_d_cost_data_real':  1.0,
            'lb_d_cost_data_fake':  0.5,
            'lb_d_cost_gen_real':   0.5,
            'learning_rate':        0.0002,
            'learning_rate_epoch':  10,
            'learning_rate_adj':    0.8,
            'mb_kernel_dim':        5,
            'mb_nb_kernels':        100,
            'nb_batch_store_gpu':   25,
            'nb_epochs':            500,
            'perc_train':           0.98,
            'perc_valid':           0.00,
            'perc_test':            0.02,
            'z_dim':                256
        }

    def _get_random_hparams(self):
        # TODO: Fix random hparams selection
        hparams = {}
        return hparams
