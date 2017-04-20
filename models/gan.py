import collections
import lib
from lib.inits import constant, he, normal, orthogonal
from lib.layers import lrelu, relu, conv, conv_1d, sigmoid, deconv, tanh, batchnorm, concat, max_pool, max_pool_1d, avg_pool, dropout, BinaryCrossEntropy, MeanSquaredError, MeanAbsoluteError
from lib.rng import t_rng, np_rng
from lib.updates import Adam, Regularizer
from lib.utils import castX, shared0s, sharedX, floatX
import math
import numpy as np
import theano
import theano.tensor as T
from theano.ifelse import ifelse
from . import BaseModel
from settings import MAX_HEIGHT, MAX_WIDTH, SAMPLES_TO_GENERATE, SAVE_MODEL_TO_DISK

class GenerativeAdversarialNetwork(BaseModel):
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
        lb_l2_reg = self.hparams.get('lb_l2_reg')
        lb_g_cost_disc = self.hparams.get('lb_g_cost_disc')
        lb_g_border_cost = self.hparams.get('lb_g_border_cost')
        lb_g_feature_matching = self.hparams.get('lb_g_feature_matching')
        lb_g_dummy_vbatchnorm = self.hparams.get('lb_g_dummy_vbatchnorm')
        lb_g_avg_cost = self.hparams.get('lb_g_avg_cost')
        lb_d_cost_data_real = self.hparams.get('lb_d_cost_data_real')
        lb_d_cost_data_fake = self.hparams.get('lb_d_cost_data_fake')
        lb_d_cost_gen_real = self.hparams.get('lb_d_cost_gen_real')
        lb_d_avg_cost = self.hparams.get('lb_d_avg_cost')
        adam_b1 = self.hparams.get('adam_b1')
        emb_adam_b1 = self.hparams.get('emb_adam_b1')
        mask_min_perc, mask_max_perc = self.hparams.get('mask_min_perc', 0.3), self.hparams.get('mask_max_perc', 0.85)
        hist_alpha = 1. / max(100., 2. * nb_batch_train)

        # GAN
        gen_filters = self.hparams.get('gen_filters')                               # Number of filters in generator
        disc_filters = self.hparams.get('disc_filters')                             # Number of filters in discriminator
        z_dim = self.hparams.get('z_dim')                                           # Noise dimensions
        perc_alpha = self.hparams.get('perc_alpha')                                 # % of conditional to apply at each gen level
        mb_nb_kernels = self.hparams.get('mb_nb_kernels')                           # Number of kernels for mini-batch discrimination
        mb_kernel_dim = self.hparams.get('mb_kernel_dim')                           # Dimension of each mini-batch kernel

        # SqueezeNet
        squ_dim = self.hparams.get('squ_dim')                                       # SqueezNet dims
        squ_dropout = self.hparams.get('squ_dropout', 0.5)

        # Embedding
        emb_cnn_dim = self.hparams.get('emb_cnn_dim')
        emb_dim = self.hparams.get('emb_dim')
        emb_out_dim = self.hparams.get('emb_out_dim')
        emb_doc_length = self.hparams.get('emb_doc_length')

        # Fixed
        input_maps = 3  # 3 channels
        emb_alphabet = 'abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:\'"/\\|_@#$%^&*~`+-=<>()[]{} '
        emb_char_to_ix = {ch: ix for ix, ch in enumerate(emb_alphabet)}
        emb_alphabet_size = len(emb_alphabet)

        # Theano symbolic variables
        index = T.iscalar('index')
        in_training = T.bscalar('in_training')                                      # pseudo boolean for switching between training and prediction
        mask_already_cropped = T.bscalar('mask_already_cropped')                    # pseudo boolean
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
        self.reference_batch = shared0s(shape=(batch_size, input_maps, MAX_HEIGHT, MAX_WIDTH), target='dev1')

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

        # SqueezeNet Parameters
        # Reference: SqueezeNet: AlexNet-Level Accuracy with 50x Fewer Parameters and <1MB Model Size, Iandola et al. 2016
        #            Exploration of the Effect of Residual Connection on top of SqueezeNet. Shen and Han.
        s_03, e_03 = 16, 64
        s_04, e_04 = 16, 64
        s_05, e_05 = 32, 128
        s_07, e_07 = 32, 128
        s_08, e_08 = 48, 192
        s_09, e_09 = 48, 192
        s_10, e_10 = 64, 256
        s_12, e_12 = 64, 256

        self.tparams['squ_01_conv'] = he(shape=(96, input_maps, 3, 3), fan_in=(96 * 5 * 5), name='squ_01_conv', target='dev1')     # b, 96, 64, 64
        self.tparams['squ_01_g'] = normal(shape=(96,), mean=1.0, std_dev=0.02, name='squ_01_g', target='dev1')
        self.tparams['squ_01_b'] = constant(shape=(96,), c=0., name='squ_01_b', target='dev1')
        self.tparams['squ_03_w11s'] = he(shape=(s_03, 96, 1, 1), fan_in=(s_03 * 1 * 1), name='squ_03_w11s', target='dev1')         # b, s_03, 32, 32
        self.tparams['squ_03_w11e'] = he(shape=(e_03, s_03, 1, 1), fan_in=(e_03 * 1 * 1), name='squ_03_w11e', target='dev1')       # b, e_03, 32, 32
        self.tparams['squ_03_w33e'] = he(shape=(e_03, s_03, 3, 3), fan_in=(e_03 * 3 * 3), name='squ_03_w33e', target='dev1')       # b, e_03, 32, 32
        self.tparams['squ_03_w11b'] = he(shape=(2 * e_03, 96, 1, 1), fan_in=(2 * e_03 * 1 * 1), name='squ_03_w11b', target='dev1') # b, 2 * e_03, 32, 32
        self.tparams['squ_03_g'] = normal(shape=(2 * e_03,), mean=1.0, std_dev=0.02, name='squ_03_g', target='dev1')
        self.tparams['squ_03_b'] = constant(shape=(2 * e_03,), c=0., name='squ_03_b', target='dev1')
        self.tparams['squ_04_w11s'] = he(shape=(s_04, 2 * e_03, 1, 1), fan_in=(s_04 * 1 * 1), name='squ_04_w11s', target='dev1')   # b, s_03, 32, 32
        self.tparams['squ_04_w11e'] = he(shape=(e_04, s_04, 1, 1), fan_in=(e_04 * 1 * 1), name='squ_04_w11e', target='dev1')
        self.tparams['squ_04_w33e'] = he(shape=(e_04, s_04, 3, 3), fan_in=(e_04 * 3 * 3), name='squ_03_w33e', target='dev1')
        self.tparams['squ_04_g'] = normal(shape=(2 * e_04,), mean=1.0, std_dev=0.02, name='squ_04_g', target='dev1')
        self.tparams['squ_04_b'] = constant(shape=(2 * e_04,), c=0., name='squ_04_b', target='dev1')
        self.tparams['squ_05_w11s'] = he(shape=(s_05, 2 * e_04, 1, 1), fan_in=(s_05 * 1 * 1), name='squ_05_w11s', target='dev1')
        self.tparams['squ_05_w11e'] = he(shape=(e_05, s_05, 1, 1), fan_in=(e_05 * 1 * 1), name='squ_05_w11e', target='dev1')
        self.tparams['squ_05_w33e'] = he(shape=(e_05, s_05, 3, 3), fan_in=(e_05 * 3 * 3), name='squ_05_w33e', target='dev1')
        self.tparams['squ_05_w11b'] = he(shape=(2 * e_05, 2 * e_04, 1, 1), fan_in=(2 * e_05 * 1 * 1), name='squ_05_w11b', target='dev1')
        self.tparams['squ_05_g'] = normal(shape=(2 * e_05,), mean=1.0, std_dev=0.02, name='squ_05_g', target='dev1')
        self.tparams['squ_05_b'] = constant(shape=(2 * e_05,), c=0., name='squ_05_b', target='dev1')
        self.tparams['squ_07_w11s'] = he(shape=(s_07, 2 * e_05, 1, 1), fan_in=(s_07 * 1 * 1), name='squ_07_w11s', target='dev1')
        self.tparams['squ_07_w11e'] = he(shape=(e_07, s_07, 1, 1), fan_in=(e_07 * 1 * 1), name='squ_07_w11e', target='dev1')
        self.tparams['squ_07_w33e'] = he(shape=(e_07, s_07, 3, 3), fan_in=(e_07 * 3 * 3), name='squ_07_w33e', target='dev1')
        self.tparams['squ_07_g'] = normal(shape=(2 * e_07,), mean=1.0, std_dev=0.02, name='squ_07_g', target='dev1')
        self.tparams['squ_07_b'] = constant(shape=(2 * e_07,), c=0., name='squ_07_b', target='dev1')
        self.tparams['squ_08_w11s'] = he(shape=(s_08, 2 * e_07, 1, 1), fan_in=(s_08 * 1 * 1), name='squ_08_w11s', target='dev1')
        self.tparams['squ_08_w11e'] = he(shape=(e_08, s_08, 1, 1), fan_in=(e_08 * 1 * 1), name='squ_08_w11e', target='dev1')
        self.tparams['squ_08_w33e'] = he(shape=(e_08, s_08, 3, 3), fan_in=(e_08 * 3 * 3), name='squ_03_w33e', target='dev1')
        self.tparams['squ_08_w11b'] = he(shape=(2 * e_08, 2 * e_07, 1, 1), fan_in=(2 * e_08 * 1 * 1), name='squ_08_w11b', target='dev1')
        self.tparams['squ_08_g'] = normal(shape=(2 * e_08,), mean=1.0, std_dev=0.02, name='squ_08_g', target='dev1')
        self.tparams['squ_08_b'] = constant(shape=(2 * e_08,), c=0., name='squ_08_b', target='dev1')
        self.tparams['squ_09_w11s'] = he(shape=(s_09, 2 * e_08, 1, 1), fan_in=(s_09 * 1 * 1), name='squ_09_w11s', target='dev1')
        self.tparams['squ_09_w11e'] = he(shape=(e_09, s_09, 1, 1), fan_in=(e_09 * 1 * 1), name='squ_09_w11e', target='dev1')
        self.tparams['squ_09_w33e'] = he(shape=(e_09, s_09, 3, 3), fan_in=(e_09 * 3 * 3), name='squ_09_w33e', target='dev1')
        self.tparams['squ_09_g'] = normal(shape=(2 * e_09,), mean=1.0, std_dev=0.02, name='squ_09_g', target='dev1')
        self.tparams['squ_09_b'] = constant(shape=(2 * e_09,), c=0., name='squ_09_b', target='dev1')
        self.tparams['squ_10_w11s'] = he(shape=(s_10, 2 * e_09, 1, 1), fan_in=(s_10 * 1 * 1), name='squ_10_w11s', target='dev1')
        self.tparams['squ_10_w11e'] = he(shape=(e_10, s_10, 1, 1), fan_in=(e_10 * 1 * 1), name='squ_10_w11e', target='dev1')
        self.tparams['squ_10_w33e'] = he(shape=(e_10, s_10, 3, 3), fan_in=(e_10 * 3 * 3), name='squ_10_w33e', target='dev1')
        self.tparams['squ_10_w11b'] = he(shape=(2 * e_10, 2 * e_09, 1, 1), fan_in=(2 * e_10 * 1 * 1), name='squ_10_w11b', target='dev1')
        self.tparams['squ_10_g'] = normal(shape=(2 * e_10,), mean=1.0, std_dev=0.02, name='squ_10_g', target='dev1')
        self.tparams['squ_10_b'] = constant(shape=(2 * e_10,), c=0., name='squ_10_b', target='dev1')
        self.tparams['squ_12_w11s'] = he(shape=(s_12, 2 * e_10, 1, 1), fan_in=(s_12 * 1 * 1), name='squ_12_w11s', target='dev1')
        self.tparams['squ_12_w11e'] = he(shape=(e_12, s_12, 1, 1), fan_in=(e_12 * 1 * 1), name='squ_12_w11e', target='dev1')
        self.tparams['squ_12_w33e'] = he(shape=(e_12, s_12, 3, 3), fan_in=(e_12 * 3 * 3), name='squ_12_w33e', target='dev1')
        self.tparams['squ_12_g'] = normal(shape=(2 * e_12,), mean=1.0, std_dev=0.02, name='squ_12_g', target='dev1')
        self.tparams['squ_12_b'] = constant(shape=(2 * e_12,), c=0., name='squ_12_b', target='dev1')
        self.tparams['squ_13_conv'] = he(shape=(200, 2 * e_12, 1, 1), fan_in=(200 * 1 * 1), name='squ_13_conv', target='dev1')
        self.tparams['squ_13_g'] = normal(shape=(200,), mean=1.0, std_dev=0.02, name='squ_13_g', target='dev1')
        self.tparams['squ_13_b'] = constant(shape=(200,), c=0., name='squ_13_b', target='dev1')

        # Generator Parameters
        self.tparams['gen_00_squ_W'] = he(shape=(200, squ_dim), fan_in=(200), name='gen_00_squ_W', target='dev1')
        self.tparams['gen_00_squ_b'] = constant(shape=(squ_dim,), c=0., name='gen_00_squ_b', target='dev1')
        self.tparams['gen_00_emb_W'] = he(shape=(emb_dim, emb_out_dim), fan_in=(emb_dim), name='gen_00_emb_W', target='dev1')
        self.tparams['gen_00_emb_b'] = constant(shape=(emb_out_dim,), c=0., name='gen_00_emb_b', target='dev1')
        self.tparams['gen_01_W'] = he(shape=(z_dim + squ_dim + emb_out_dim, gen_filters * 8 * 4 * 4), fan_in=(z_dim + squ_dim + emb_out_dim), name='gen_01_W', target='dev1')
        self.tparams['gen_01_g'] = normal(shape=(gen_filters * 8 * 4 * 4,), mean=1.0, std_dev=0.02, name='gen_01_g', target='dev1')
        self.tparams['gen_01_b'] = constant(shape=(gen_filters * 8 * 4 * 4,), c=0., name='gen_01_b', target='dev1')
        self.tparams['gen_02_conv'] = he(shape=(gen_filters * 8, gen_filters * 4, 5, 5), fan_in=(gen_filters * 8 * 5 * 5), name='gen_02_conv', target='dev1')
        self.tparams['gen_02_g'] = normal(shape=(gen_filters * 4,), mean=1.0, std_dev=0.02, name='gen_02_g', target='dev1')
        self.tparams['gen_02_b'] = constant(shape=(gen_filters * 4,), c=0., name='gen_02_b', target='dev1')
        self.tparams['gen_03_conv'] = he(shape=(gen_filters * 4, gen_filters * 2, 5, 5), fan_in=(gen_filters * 4 * 5 * 5), name='gen_03_conv', target='dev1')
        self.tparams['gen_03_g'] = normal(shape=(gen_filters * 2,), mean=1.0, std_dev=0.02, name='gen_03_g', target='dev1')
        self.tparams['gen_03_b'] = constant(shape=(gen_filters * 2,), c=0., name='gen_03_b', target='dev1')
        self.tparams['gen_04_conv'] = he(shape=(gen_filters * 2, gen_filters, 5, 5), fan_in=(gen_filters * 2 * 5 * 5), name='gen_04_conv', target='dev1')
        self.tparams['gen_04_g'] = normal(shape=(gen_filters,), mean=1.0, std_dev=0.02, name='gen_04_g', target='dev1')
        self.tparams['gen_04_b'] = constant(shape=(gen_filters,), c=0., name='gen_04_b', target='dev1')
        self.tparams['gen_05_conv'] = he(shape=(gen_filters, input_maps, 5, 5), fan_in=(gen_filters * 5 * 5), name='gen_05_conv', target='dev1')
        gen_02_X_shape = (None, gen_filters * 8, 4, 4)
        gen_03_X_shape = (None, gen_filters * 4, 8, 8)
        gen_04_X_shape = (None, gen_filters * 2, 16, 16)
        gen_05_X_shape = (None, gen_filters * 1, 32, 32)

        # Generator - Historical Averaging
        list_gen_params = [param for param in self.tparams.keys() if param.startswith('gen_')]
        g_avg_updates =[]
        for gen_param in list_gen_params:
            self.tparams['avg_'+gen_param] = shared0s(self.tparams[gen_param].get_value().shape, target='dev1')
            g_avg_updates.append((self.tparams['avg_'+gen_param], hist_alpha * self.tparams[gen_param] + (1. - hist_alpha) * self.tparams['avg_'+gen_param]))

        # Discriminator Parameters
        self.tparams['disc_00_emb_W'] = he(shape=(emb_dim, emb_out_dim), fan_in=(emb_dim), name='disc_00_emb_W', target='dev0')
        self.tparams['disc_00_emb_g'] = normal(shape=(emb_out_dim,), mean=1.0, std_dev=0.02, name='disc_00_emb_g', target='dev0')
        self.tparams['disc_00_emb_b'] = constant(shape=(emb_out_dim,), c=0., name='disc_00_emb_b', target='dev0')
        self.tparams['disc_01_conv'] = he(shape=(disc_filters, input_maps, 5, 5), fan_in=(input_maps * 5 * 5), name='disc_01_conv', target='dev0')
        self.tparams['disc_02_conv'] = he(shape=(disc_filters * 2, disc_filters, 5, 5), fan_in=(disc_filters * 5 * 5), name='disc_02_conv', target='dev0')
        self.tparams['disc_02_g'] = normal(shape=(disc_filters * 2,), mean=1.0, std_dev=0.02, name='disc_02_g', target='dev0')
        self.tparams['disc_02_b'] = constant(shape=(disc_filters * 2,), c=0., name='disc_02_b', target='dev0')
        self.tparams['disc_03_conv'] = he(shape=(disc_filters * 4, disc_filters * 2, 5, 5), fan_in=(disc_filters * 2 * 5 * 5), name='disc_03_conv', target='dev0')
        self.tparams['disc_03_g'] = normal(shape=(disc_filters * 4,), mean=1.0, std_dev=0.02, name='disc_03_g', target='dev0')
        self.tparams['disc_03_b'] = constant(shape=(disc_filters * 4,), c=0., name='disc_03_b', target='dev0')
        self.tparams['disc_04_conv'] = he(shape=(disc_filters * 8, disc_filters * 4, 5, 5), fan_in=(disc_filters * 4 * 5 * 5), name='disc_04_conv', target='dev0')
        self.tparams['disc_04_mb_W'] = he(shape=(disc_filters * 8 * 4 * 4, mb_nb_kernels * mb_kernel_dim), fan_in=(disc_filters * 8 * 4 * 4), name='disc_04_mb_W', target='dev0')
        self.tparams['disc_04_mb_b'] = constant(shape=(mb_nb_kernels,), c=0., name='disc_04_mb_b', target='dev0')
        self.tparams['disc_05_g'] = normal(shape=(disc_filters * 8 + emb_out_dim + mb_nb_kernels,), mean=1.0, std_dev=0.02, name='disc_05_g', target='dev0')
        self.tparams['disc_05_b'] = constant(shape=(disc_filters * 8 + emb_out_dim + mb_nb_kernels,), c=0., name='disc_05_b', target='dev0')
        self.tparams['disc_06_W'] = he(shape=((disc_filters * 8 + emb_out_dim + mb_nb_kernels) * 4 * 4, 1), fan_in=((disc_filters * 8 + emb_out_dim + mb_nb_kernels) * 4 * 4), name='disc_06_W', target='dev0')
        self.tparams['disc_06_b'] = constant(shape=(1,), c=0., name='disc_06_b', target='dev0')

        # Discriminator - Historical Averaging
        list_disc_params = [param for param in self.tparams.keys() if param.startswith('disc_')]
        d_avg_updates =[]
        for disc_param in list_disc_params:
            self.tparams['avg_'+disc_param] = shared0s(self.tparams[disc_param].get_value().shape, target='dev0')
            d_avg_updates.append((self.tparams['avg_'+disc_param], hist_alpha * self.tparams[disc_param] + (1. - hist_alpha) * self.tparams['avg_'+disc_param]))


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
            o_h1 = max_pool_1d(o_h1, ws=3, stride=3)                                                    # (batch, 384, 66)
            o_h2 = relu(conv_1d(o_h1, tp['emb_cnn_02_conv'], subsample=(1,), border_mode='valid'))  # (batch, 512, 63) (kernel 512 x 384 x 4)
            o_h2 = max_pool_1d(o_h2, ws=3, stride=3)                                                    # (batch, 512, 21)
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
            l_h1 = T.mean(rval[0], axis=0, dtype=theano.config.floatX, acc_dtype=theano.config.floatX)  # (batch, cnn_dim) - Embedding is the average of h_t, and not the final h_t
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
        # SqueezeNet fire modules (with bypass)
        # ---------------------
        def fire_type_A(input, w11s, w11e, w33e, w11b, g, b):
            squ_1x1 = conv(input, w11s, subsample=(1, 1), border_mode='half')
            exp_1x1 = conv(squ_1x1, w11e, subsample=(1, 1), border_mode='half')
            exp_3x3 = conv(squ_1x1, w33e, subsample=(1, 1), border_mode='half')
            exp_con = concat([exp_1x1, exp_3x3], axis=1)
            byp_1x1 = conv(input, w11b, subsample=(1, 1), border_mode='half')
            out     = relu(exp_con + byp_1x1)
            return batchnorm(out, g=g, b=b)

        def fire_type_B(input, w11s, w11e, w33e, g, b):
            squ_1x1 = conv(input, w11s, subsample=(1, 1), border_mode='half')
            exp_1x1 = conv(squ_1x1, w11e, subsample=(1, 1), border_mode='half')
            exp_3x3 = conv(squ_1x1, w33e, subsample=(1, 1), border_mode='half')
            exp_con = concat([exp_1x1, exp_3x3], axis=1)
            out     = relu(exp_con + input)
            return batchnorm(out, g=g, b=b)

        # SqueezeNet
        def squeeze_net(X, already_cropped, tp):
            s_h0  = castX((T.zeros_like(src_images) * mask) + (1. - mask) * src_images + mask_color)             # Image with mask cropped
            s_h0  = ifelse(already_cropped, X, s_h0)                                                             # b, 3, 64, 64
            s_h1  = conv(s_h0, tp['squ_01_conv'], subsample=(1, 1), border_mode='half')                          # b, 96, 64, 64
            s_h1  = batchnorm(s_h1, g=tp['squ_01_g'], b=tp['squ_01_b'])
            s_h2  = max_pool(s_h1, ws=(2, 2))                                                                    # b, 96, 32, 32
            s_h3  = fire_type_A(s_h2, *[tp['squ_03_' + k] for k in ['w11s', 'w11e', 'w33e', 'w11b', 'g', 'b']])  # b, 128, 32, 32
            s_h4  = fire_type_B(s_h3, *[tp['squ_04_' + k] for k in ['w11s', 'w11e', 'w33e', 'g', 'b']])          # b, 128, 32, 32
            s_h5  = fire_type_A(s_h4, *[tp['squ_05_' + k] for k in ['w11s', 'w11e', 'w33e', 'w11b', 'g', 'b']])  # b, 256, 32, 32
            s_h6  = max_pool(s_h5, ws=(2, 2))                                                                    # b, 256, 16, 16
            s_h7  = fire_type_B(s_h6, *[tp['squ_07_' + k] for k in ['w11s', 'w11e', 'w33e', 'g', 'b']])          # b, 256, 16, 16
            s_h8  = fire_type_A(s_h7, *[tp['squ_08_' + k] for k in ['w11s', 'w11e', 'w33e', 'w11b', 'g', 'b']])  # b, 384, 16, 16
            s_h9  = fire_type_B(s_h8, *[tp['squ_09_' + k] for k in ['w11s', 'w11e', 'w33e', 'g', 'b']])          # b, 384, 16, 16
            s_h10 = fire_type_A(s_h9, *[tp['squ_10_' + k] for k in ['w11s', 'w11e', 'w33e', 'w11b', 'g', 'b']])  # b, 512, 16, 16
            s_h11 = max_pool(s_h10, ws=(2, 2))                                                                   # b, 512, 8, 8
            s_h12 = fire_type_B(s_h11, *[tp['squ_12_' + k] for k in ['w11s', 'w11e', 'w33e', 'g', 'b']])         # b, 512, 8, 8
            s_h12 = ifelse(in_training, dropout(s_h12, p=squ_dropout), s_h12)                                    # b, 512, 8, 8
            s_h13 = conv(s_h12, tp['squ_13_conv'], subsample=(1, 1), border_mode='half')                         # b, 200, 8, 8
            s_h13 = batchnorm(s_h13, g=tp['squ_13_g'], b=tp['squ_13_b'])
            s_x   = avg_pool(s_h13, ws=(8, 8)).flatten(2)                                                        # b, 200
            return s_x

        # ---------------------
        # Generator
        # ---------------------
        def repeat_colors(layer, repeat_red, repeat_green, repeat_blue):
            return concat([
                T.repeat(layer[:, 0, None, :, :], repeat_red, axis=1),
                T.repeat(layer[:, 1, None, :, :], repeat_green, axis=1),
                T.repeat(layer[:, 2, None, :, :], repeat_blue, axis=1)], axis=1)

        def generator(Z, s_x, emb, tp):
            g_h0_mask = mask                                                                                                                # b, 1, 64, 64
            g_h0_s = lrelu(T.dot(s_x, tp['gen_00_squ_W']) + tp['gen_00_squ_b'])                                                             # b. squ_dim (128)
            g_h0_e = lrelu(T.dot(emb, tp['gen_00_emb_W']) + tp['gen_00_emb_b'])                                                             # b, emb_out_dim (128)
            g_h0_b = castX((T.zeros_like(src_images) * mask) + (1. - mask) * src_images + mask_color)                                       # b, 3, 64, 64
            g_h0 = concat([Z, g_h0_s, g_h0_e], axis=1)                                                                                      # b, 100 + 128 + 128
            g_h1 = lrelu(batchnorm(T.dot(g_h0, tp['gen_01_W']), g=tp['gen_01_g'], b=tp['gen_01_b']))                                        # b, 16384
            g_h1 = g_h1.reshape(shape=(current_batch_size, gen_filters * 8, 4, 4))                                                          # b, 1024, 4, 4
            g_h1_b = repeat_colors(max_pool(g_h0_b, ws=(16, 16)), 341, 341, 342)                                                            # b, 1024, 4, 4
            g_h1_o = perc_alpha * g_h1 + (1. - perc_alpha) * g_h1_b                                                                         # b, 1024, 4, 4
            g_h2 = deconv(g_h1_o, gen_02_X_shape, w=tp['gen_02_conv'], subsample=(2,2), border_mode=(2, 2), target_size=(8, 8))             # b, 512, 8, 8
            g_h2 = lrelu(batchnorm(g_h2, g=tp['gen_02_g'], b=tp['gen_02_b']))                                                               # b, 512, 8, 8
            g_h2_b = repeat_colors(max_pool(g_h0_b, ws=(8, 8)), 170, 170, 172)                                                              # b, 512, 8, 8
            g_h2_o = perc_alpha * g_h2 + (1. - perc_alpha) * g_h2_b                                                                         # b, 512, 8, 8
            g_h3 = deconv(g_h2_o, gen_03_X_shape, w=tp['gen_03_conv'], subsample=(2, 2), border_mode=(2, 2), target_size=(16, 16))          # b, 256, 16, 16
            g_h3 = lrelu(batchnorm(g_h3, g=tp['gen_03_g'], b=tp['gen_03_b']))                                                               # b, 256, 16, 16
            g_h3_b = repeat_colors(max_pool(g_h0_b, ws=(4, 4)), 85, 85, 86)                                                                 # b, 256, 16, 16
            g_h3_o = perc_alpha * g_h3 + (1. - perc_alpha) * g_h3_b                                                                         # b, 256, 16, 16
            g_h4 = deconv(g_h3_o, gen_04_X_shape, w=tp['gen_04_conv'], subsample=(2, 2), border_mode=(2, 2), target_size=(32, 32))          # b, 128, 32, 32
            g_h4 = lrelu(batchnorm(g_h4, g=tp['gen_04_g'], b=tp['gen_04_b']))                                                               # b, 128, 32, 32
            g_h4_b = repeat_colors(max_pool(g_h0_b, ws=(2, 2)), 42, 42, 44)                                                                 # b, 128, 32, 32
            g_h4_o = perc_alpha * g_h4 + (1. - perc_alpha) * g_h4_b                                                                         # b, 128, 32, 32
            g_h5 = deconv(g_h4_o, gen_05_X_shape, w=tp['gen_05_conv'], subsample=(2, 2), border_mode=(2, 2), target_size=(64, 64))          # b, 3, 64, 64
            g_x = tanh(g_h5) * g_h0_mask + (1. - g_h0_mask) * ifelse(mask_already_cropped, src_images, src_images * (1. - mask))
            return g_x

        # ---------------------
        # Discriminator
        # ---------------------
        # Mini-batch discrimination
        # Adapted from https://github.com/openai/improved-gan/blob/master/mnist_svhn_cifar10/nn.py#L132
        def minibatch(d_h4, W, b):
            mb_h0 = T.dot(d_h4.flatten(2), W)                                                           # b, nb_kernels * kernel_dim
            mb_h0 = mb_h0.reshape((d_h4.shape[0], mb_nb_kernels, mb_kernel_dim))                        # b, nb_kernel, kernel_dim
            mb_h1 = mb_h0.dimshuffle(0, 1, 2, 'x') - mb_h0.dimshuffle('x', 1, 2, 0)
            mb_h1 = T.sum(abs(mb_h1), axis=2) + 1e6 * T.eye(d_h4.shape[0]).dimshuffle(0,'x',1)
            mb_h2 = T.sum(T.exp(-mb_h1), axis=2) + b                                                    # b, nb_kernel
            mb_h2 = mb_h2.dimshuffle(0, 1, 'x', 'x')
            mb_h2 = T.repeat(mb_h2, 4, axis=2)
            mb_h2 = T.repeat(mb_h2, 4, axis=3)
            return mb_h2

        def discriminator(g_x, emb, tp):
            d_h0_e = T.dot(emb, tp['disc_00_emb_W'])                                                    # b, 128
            d_h0_e = lrelu(batchnorm(d_h0_e, g=tp['disc_00_emb_g'], b=tp['disc_00_emb_b'])).dimshuffle(0, 1, 'x', 'x')
            d_h0_e = T.repeat(d_h0_e, 4, axis=2)
            d_h0_e = T.repeat(d_h0_e, 4, axis=3)                                                        # b, 128, 4, 4
            d_h0_b = g_x * mask + (1. - mask) * (src_images + g_x) / 2.                                 # b, 3, 64, 64 - Border is avg of src and g_x
            d_h0 = ifelse(mask_already_cropped, g_x, d_h0_b)                                            # (looks pixelated if border has been changed by generator)
            d_h1 = lrelu(conv(d_h0, tp['disc_01_conv'], subsample=(2, 2), border_mode=(2, 2)))          # b, 128, 32, 32
            d_h2 = conv(d_h1, tp['disc_02_conv'], subsample=(2, 2), border_mode=(2, 2))                 # b, 256, 16, 16
            d_h2 = lrelu(batchnorm(d_h2, g=tp['disc_02_g'], b=tp['disc_02_b']))
            d_h3 = conv(d_h2, tp['disc_03_conv'], subsample=(2, 2), border_mode=(2, 2))                 # b, 512, 8, 8
            d_h3 = lrelu(batchnorm(d_h3, g=tp['disc_03_g'], b=tp['disc_03_b']))
            d_h4 = conv(d_h3, tp['disc_04_conv'], subsample=(2, 2), border_mode=(2, 2))                 # b, 1024, 4, 4
            d_h4_mb = minibatch(d_h4, W=tp['disc_04_mb_W'], b=tp['disc_04_mb_b'])
            d_h5 = concat([d_h4, d_h0_e, d_h4_mb], axis=1)                                              # b, 1024+128+nb_kernel, 4, 4
            d_h5 = lrelu(batchnorm(d_h5, g=tp['disc_05_g'], b=tp['disc_05_b']))
            d_h5 = T.flatten(d_h5, 2)                                                                   # b, 18432
            d_y  = sigmoid(T.dot(d_h5, tp['disc_06_W']) + tp['disc_06_b'])                              # b, 1
            return d_y, d_h3

        # ---------------------
        # Cost and updates
        # ---------------------
        # Embedding cost and updates
        emb_1 = document_lstm(document_conv_net(one_hot_1, self.tparams), self.tparams)         # Real caption
        emb_2 = document_lstm(document_conv_net(one_hot_2, self.tparams), self.tparams)         # Matching caption (for embedding training), Fake caption (for disc. training)
        emb_loss, emb_accuracy = joint_embedding_loss(emb_1, emb_2)
        emb_params = [self.tparams[k] for k in self.tparams.keys() if k.startswith('emb_')]
        emb_updater = Adam(lr=self.tparams['hyper_emb_learning_rate'], b1=emb_adam_b1, clipnorm=10.)
        emb_updates = emb_updater(emb_params, emb_loss)

        # Different scenarios (real data, fake data)
        ref_s_x = squeeze_net(self.reference_batch[0:current_batch_size], 1, self.tparams)      # Virtual batch normalization (squeeze net)
        ref_g_x = generator(z, ref_s_x, emb_1, self.tparams)                                    # Virtual batch normalization (generator)
        s_x = squeeze_net(src_images, mask_already_cropped, self.tparams)
        g_x = generator(z, s_x, emb_1, self.tparams)
        p_data_real, data_real_h3 = discriminator(src_images, emb_1, self.tparams)
        p_data_fake, data_fake_h3 = discriminator(src_images, emb_2, self.tparams)
        p_gen_real, gen_real_h3  = discriminator(g_x, emb_1, self.tparams)

        # Historical Averaging
        g_avg_cost, d_avg_cost = 0., 0.
        nb_g_avg_param, nb_d_avg_param = 0., 0.
        for param in self.tparams.keys():
            if param.startswith('gen_'):
                g_avg_cost += MeanSquaredError(self.tparams[param], self.tparams['avg_'+param])
                nb_g_avg_param += 1
            if param.startswith('disc_'):
                d_avg_cost += MeanSquaredError(self.tparams[param], self.tparams['avg_'+param])
                nb_d_avg_param += 1
        g_avg_cost = g_avg_cost / max(1., nb_g_avg_param)
        d_avg_cost = d_avg_cost / max(1., nb_d_avg_param)

        # Generator cost and updates
        g_cost_disc = BinaryCrossEntropy(p_gen_real, 0.9)
        g_border_cost = MeanSquaredError(g_x * (1. - mask), src_images * (1. - mask))
        g_feature_matching = MeanSquaredError(gen_real_h3, data_real_h3)                                    # Feature matching (d_h3 of discriminator)
        g_dummy_vbatchnorm = T.clip(MeanAbsoluteError(s_x, ref_s_x[0:current_batch_size]), 0, 0.00001)      # Dummy cost to make sure virtual batch is included in graph
        g_dummy_vbatchnorm += T.clip(MeanAbsoluteError(g_x, ref_g_x[0:current_batch_size]), 0, 0.00001)
        g_total_cost  = lb_g_cost_disc * g_cost_disc
        g_total_cost += lb_g_border_cost * g_border_cost
        g_total_cost += lb_g_feature_matching * g_feature_matching
        g_total_cost += lb_g_dummy_vbatchnorm * g_dummy_vbatchnorm
        g_total_cost += lb_g_avg_cost * g_avg_cost
        g_params = [self.tparams[k] for k in self.tparams.keys() if k.startswith('gen_')]
        g_params += [self.tparams[k] for k in self.tparams.keys() if k.startswith('squ_')]
        g_updater = Adam(lr=self.tparams['hyper_learning_rate'], b1=adam_b1, clipnorm=10., regularizer=Regularizer(l2=lb_l2_reg))
        g_updates = g_updater(g_params, g_total_cost) + g_avg_updates
        g_border_updates = g_updater(g_params, g_border_cost)

        # Discriminator cost and updates
        d_cost_data_real = BinaryCrossEntropy(p_data_real, 0.9)                                 # Real image, real text (0.9 => One-Sided Label Smoothing)
        d_cost_data_fake = BinaryCrossEntropy(p_data_fake, 0.0)                                 # Real image, fake text
        d_cost_gen_real  = BinaryCrossEntropy(p_gen_real, 0.0)                                  # Generator Image, real text
        d_total_cost  = lb_d_cost_data_real * d_cost_data_real
        d_total_cost += lb_d_cost_data_fake * d_cost_data_fake
        d_total_cost += lb_d_cost_gen_real * d_cost_gen_real
        d_total_cost += lb_d_avg_cost * d_avg_cost
        d_params = [self.tparams[k] for k in self.tparams.keys() if k.startswith('disc_')]
        d_updater = Adam(lr=self.tparams['hyper_learning_rate'], b1=adam_b1, clipnorm=10., regularizer=Regularizer(l2=lb_l2_reg))
        d_updates = d_updater(d_params, d_total_cost) + d_avg_updates

        # Combined
        d_g_updates = d_updates + g_updates

        # Compilation variables
        self.cvars = {}
        cvars_list = ['index', 'src_images', 'one_hot_1', 'one_hot_2', 'mask_already_cropped', 'in_training', 'batch_size', 'emb_loss',
                      'emb_accuracy', 'emb_updates', 'z', 'g_x', 'g_total_cost', 'g_updates', 'g_border_updates', 'd_total_cost',
                      'd_updates', 'd_g_updates', 'g_cost_disc', 'g_border_cost', 'g_feature_matching', 'g_dummy_vbatchnorm',
                      'g_avg_cost', 'd_cost_data_real', 'd_cost_data_fake', 'd_cost_gen_real', 'd_avg_cost']
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
                    _['one_hot_2']: self.emb_dataset[_['index'] * _['batch_size']: (_['index'] + 1) * _['batch_size'], 1, :, :],
                    _['in_training']: np.cast['int8'](1),
                    _['mask_already_cropped']: np.cast['int8'](0)
                },
                on_unused_input='ignore')
            print('..... Done compiling train_embedding_fn()')

        if fn_name == 'train_gen' and not hasattr(self, 'train_gen_fn'):
            print('..... Compiling train_gen_fn()')
            self.train_gen_fn = theano.function(
                inputs=[_['index'], _['z']],
                outputs=[_['g_total_cost'], _['g_cost_disc'], _['g_border_cost'], _['g_feature_matching'], _['g_dummy_vbatchnorm'],
                         _['g_avg_cost']],
                updates=_['g_updates'],
                givens={
                    _['src_images']: self.gpu_dataset[_['index'] * _['batch_size']: (_['index'] + 1) * _['batch_size'], :, :, :],
                    _['one_hot_1']: self.emb_dataset[_['index'] * _['batch_size']: (_['index'] + 1) * _['batch_size'], 0, :, :],
                    _['one_hot_2']: self.emb_dataset[_['index'] * _['batch_size']: (_['index'] + 1) * _['batch_size'], 1, :, :],
                    _['in_training']: np.cast['int8'](1),
                    _['mask_already_cropped']: np.cast['int8'](0)
                },
                on_unused_input='warn')
            print('..... Done compiling train_gen_fn()')

        if fn_name == 'train_gen_border' and not hasattr(self, 'train_gen_border_fn'):
            print('..... Compiling train_gen_border_fn()')
            self.train_gen_border_fn = theano.function(
                inputs=[_['index'], _['z']],
                outputs=_['g_border_cost'],
                updates=_['g_border_updates'],
                givens={
                    _['src_images']: self.gpu_dataset[_['index'] * _['batch_size']: (_['index'] + 1) * _['batch_size'], :, :, :],
                    _['one_hot_1']: self.emb_dataset[_['index'] * _['batch_size']: (_['index'] + 1) * _['batch_size'], 0, :, :],
                    _['one_hot_2']: self.emb_dataset[_['index'] * _['batch_size']: (_['index'] + 1) * _['batch_size'], 1, :, :],
                    _['in_training']: np.cast['int8'](1),
                    _['mask_already_cropped']: np.cast['int8'](0)
                },
                on_unused_input='warn')
            print('..... Done compiling train_gen_border_fn()')

        if fn_name == 'train_disc' and not hasattr(self, 'train_disc_fn'):
            print('..... Compiling train_disc_fn()')
            self.train_disc_fn = theano.function(
                inputs=[_['index'], _['z']],
                outputs=[_['d_total_cost'], _['d_cost_data_real'], _['d_cost_data_fake'], _['d_cost_gen_real'], _['d_avg_cost']],
                updates=_['d_updates'],
                givens={
                    _['src_images']: self.gpu_dataset[_['index'] * _['batch_size']: (_['index'] + 1) * _['batch_size'], :, :, :],
                    _['one_hot_1']: self.emb_dataset[_['index'] * _['batch_size']: (_['index'] + 1) * _['batch_size'], 0, :, :],
                    _['one_hot_2']: self.emb_dataset[_['index'] * _['batch_size']: (_['index'] + 1) * _['batch_size'], 1, :, :],
                    _['in_training']: np.cast['int8'](1),
                    _['mask_already_cropped']: np.cast['int8'](0)
                },
                on_unused_input='warn')
            print('..... Done compiling train_disc_fn()')

        if fn_name == 'train_gen_and_disc' and not hasattr(self, 'train_gen_and_disc_fn'):
            print('..... Compiling train_gen_and_disc_fn()')
            self.train_gen_and_disc_fn = theano.function(
                inputs=[_['index'], _['z']],
                outputs=[_['g_total_cost'], _['d_total_cost'], _['g_cost_disc'], _['g_border_cost'], _['g_feature_matching'], _['g_dummy_vbatchnorm'],
                         _['g_avg_cost'], _['d_cost_data_real'], _['d_cost_data_fake'], _['d_cost_gen_real'], _['d_avg_cost']],
                updates=_['d_g_updates'],
                givens={
                    _['src_images']: self.gpu_dataset[_['index'] * _['batch_size']: (_['index'] + 1) * _['batch_size'], :, :, :],
                    _['one_hot_1']: self.emb_dataset[_['index'] * _['batch_size']: (_['index'] + 1) * _['batch_size'], 0, :, :],
                    _['one_hot_2']: self.emb_dataset[_['index'] * _['batch_size']: (_['index'] + 1) * _['batch_size'], 1, :, :],
                    _['in_training']: np.cast['int8'](1),
                    _['mask_already_cropped']: np.cast['int8'](0)
                },
                on_unused_input='warn')
            print('..... Done compiling train_gen_and_disc_fn()')

        if fn_name == 'generate_samples' and not hasattr(self, 'generate_samples_fn'):
            print('..... Compiling generate_samples_fn()')
            self.generate_samples_fn = theano.function(
                inputs=[_['src_images'], _['z'], _['one_hot_1'], _['mask_already_cropped']],
                outputs=_['g_x'],
                givens={
                    _['in_training']: np.cast['int8'](0)
                },
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
        z_dim = self.hparams.get('z_dim')

        learning_rate_epoch = self.hparams.get('learning_rate_epoch')
        learning_rate_adj = self.hparams.get('learning_rate_adj')
        initial_learning_rate = self.hparams.get('learning_rate')
        last_cost_g = []
        last_cost_d = []

        # Function shorthands
        save_image = lib.Image().save
        display_image = lib.Image().display

        # Looping
        self.tparams['hyper_learning_rate'].set_value(initial_learning_rate)
        iter = -1
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
                    if imgs.shape[0] < batch_size:
                        raise AssertionError("Reference batch must have at least a full batch of examples")
                    self.reference_batch.set_value(imgs[0:batch_size])
                    starting_epoch = False
                else:
                    if self.loader.buffer['train'].qsize() > 0:
                        imgs, real_captions, matching_captions, fake_captions, end_of_epoch = self.loader.buffer['train'].get()
                    else:
                        print('..... Buffer is empty. Re-using current data for another batch.')
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
                    z = np_rng.randn(current_batch_size, z_dim).astype(theano.config.floatX)
                    if border_only:
                        border_cost = self.train_gen_border_fn(ix, z)
                        last_cost_g.append(border_cost)
                        last_cost_g = last_cost_g[-1000:]
                        print('Batch %04d/%04d - Epoch %04d -- G Border Loss : %.4f - Min/Max (1000): %.4f/%.4f' %
                              (batch_ix + 1, nb_batch_train, epoch + 1, border_cost, min(last_cost_g), max(last_cost_g)))
                    else:
                        cost_g, cost_d, g_cost_disc, g_border_cost, g_feature_matching, g_dummy_vbatchnorm, g_avg_cost, \
                            d_cost_data_real, d_cost_data_fake, d_cost_gen_real, d_avg_cost = self.train_gen_and_disc_fn(ix, z)
                        last_cost_g.append(cost_g)
                        last_cost_g = last_cost_g[-1000:]
                        last_cost_d.append(cost_d)
                        last_cost_d = last_cost_d[-1000:]
                        print('Batch %04d/%04d - Epoch %04d -- Loss G/D: %.4f/%.4f - Min/Max (1000): %.4f/%.4f - %.4f/%.4f ' %
                             (batch_ix + 1, nb_batch_train, epoch + 1, cost_g, cost_d, min(last_cost_g), max(last_cost_g), min(last_cost_d), max(last_cost_d)))
                        print('GDisc) %.4f / GBord) %.4f / GFeat) %.4f / GVBat) %.4f / GAvg) %.4f  |  DReal) %.4f / DFake) %.4f / DGen) %.4f / DAvg) %.4f' %
                             (g_cost_disc, g_border_cost, g_feature_matching, g_dummy_vbatchnorm, g_avg_cost, d_cost_data_real, d_cost_data_fake, d_cost_gen_real, d_avg_cost))

                if (master_batch_ix + 1) % 5 == 0 or end_of_epoch:
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
                        z = np_rng.randn(src_images.shape[0], z_dim).astype(theano.config.floatX)
                        out_images = self.generate_samples_fn(src_images, z, curr_emb, 0)
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
            'batch_size':           8,
            'disc_filters':         128,
            'emb_adam_b1':          0.5,
            'emb_cnn_dim':          256,
            'emb_dim':              512,
            'emb_doc_length':       201,
            'emb_out_dim':          128,
            'emb_learning_rate':    0.0002,
            'emb_learning_rate_epoch':  10,
            'emb_learning_rate_adj':    0.5,
            'emb_nb_epochs':        256,
            'gen_filters':          128,
            'improvement_threshold': 0.995,
            'lb_l2_reg':            1e-5,
            'lb_g_cost_disc':       0.1,
            'lb_g_border_cost':     1.0,
            'lb_g_feature_matching':10.0,
            'lb_g_dummy_vbatchnorm':1.0,
            'lb_g_avg_cost':        1.0,
            'lb_d_cost_data_real':  1.0,
            'lb_d_cost_data_fake':  0.5,
            'lb_d_cost_gen_real':   0.5,
            'lb_d_avg_cost':        1.0,
            'learning_rate':        0.0002,
            'learning_rate_epoch':  1,
            'learning_rate_adj':    0.8,
            'mask_min_perc':        0.3,
            'mask_max_perc':        0.85,
            'mb_kernel_dim':        4,
            'mb_nb_kernels':        64,
            'nb_batch_store_gpu':   1,
            'nb_epochs':            1,
            'patience':             5,
            'patience_increase':    2.,
            'perc_alpha':           0.5,
            'perc_train':           0.85,
            'perc_valid':           0.10,
            'perc_test':            0.05,
            'squ_dim':              64,
            'squ_dropout':          0.,
            'z_dim':                64
        }

    def _get_random_hparams(self):
        # TODO: Fix random hparams selection
        hparams = {}
        return hparams
