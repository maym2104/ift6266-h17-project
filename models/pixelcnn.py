"""
About:  PixelCNN implementation for the IFT6266 project
Author: Mariane Maynard
Date:   April 2017
Other:  Inspired from the tensorflow implementation of Sherjil Ozair: https://github.com/sherjilozair/ift6266/blob/master/pixelcnn.py
"""

import sys, os
import numpy as np
from lib.layers import lrelu, relu, conv, conv_1d, sigmoid, deconv, tanh, batchnorm, concat, max_pool, max_pool_1d, avg_pool, dropout, BinaryCrossEntropy, MeanSquaredError, MeanAbsoluteError
from lib.inits import constant, he, normal, orthogonal, uniform, glorot
from lib.utils import castX, shared0s, sharedX, floatX
from lib.updates import Adam, Regularizer
from PIL import Image
import theano
import theano.tensor as T
from . import BaseModel

class nn:

    @staticmethod
    def conv2d(inputs, num_outputs, kernel_size, params, mask=None, activation_fn=relu, scope=None, input_shape=None):
        shape = [num_outputs, inputs.shape[1]] + kernel_size

        W = params[scope+'_W']
        b = params[scope+'_b']

        if mask:
            mid_x = shape[2]/2
            mid_y = shape[3]/2
            mask_filter = T.ones(shape, dtype='float32')
            mask_filter = T.set_subtensor(mask_filter[:,:,mid_x, mid_y+1:], 0.)
            mask_filter = T.set_subtensor(mask_filter[:,:,mid_x+1:, :], 0.)

            if mask == 'a':
                mask_filter = T.set_subtensor(mask_filter[:,:,mid_x, mid_y], 0.)
                
            W  = W * mask_filter

        h = conv(inputs, W, b, border_mode=(kernel_size[0]/2,kernel_size[1]/2), subsample=(1,1))

        if activation_fn:
            h = activation_fn(h)

        return h

    @staticmethod
    def residual_block(x, name, params, mask=None, h_size=0):
        h = nn.conv2d(x, h_size, [1, 1], params, mask=mask, scope='conv_1x1_{}_1'.format(name))
        h = nn.conv2d(h, h_size, [3, 3], params, mask=mask, scope='conv_3x3_{}'.format(name))
        h = nn.conv2d(h, 2*h_size, [1, 1], params, mask=mask, scope='conv_1x1_{}_2'.format(name))
        return h + x


class PixelCNN(BaseModel):
    def build(self):
        self.image = T.tensor4('src_images', dtype='float32')
        input = self.image.dimshuffle(0,3,1,2)
        self.init_tparams()
        nb_h = self.hparams['h']
        nb_relu_units = self.hparams['nb_relu_units']
        h = nn.conv2d(input, 2*nb_h, [7, 7], self.tparams, mask='a', scope='conv_7x7')
        
        for i in xrange(self.hparams['nb_residual_block']):
            h = nn.residual_block(h, str(i), self.tparams, mask='b', h_size=nb_h)
        h1 = nn.conv2d(h, nb_relu_units, [1, 1], self.tparams, mask='b', scope='conv_relu_1x1_1')
        h2 = nn.conv2d(h1, nb_relu_units, [1, 1], self.tparams, mask='b', scope='conv_relu_1x1_2')
        self.logits = nn.conv2d(h2, 3, [1, 1], self.tparams, activation_fn=sigmoid, scope='conv_logits')
        self.logits = self.logits.dimshuffle(0,2,3,1)
        self.losses = T.nnet.binary_crossentropy(self.logits, self.image)
        self.losses = self.losses[:, 16:48, 16:48, :]
        self.loss = self.losses.sum(axis=[1, 2, 3]).mean()
        self.updater = Adam(self.hparams['adam_leanring_rate'])
        self.params = [self.tparams[k] for k in self.tparams.keys()]
        self.updates = self.updater(self.params, self.loss)
        self.train_op = theano.function(inputs=[self.image],outputs=[self.loss],updates=self.updates)
        self.val_op = theano.function(inputs=[self.image],outputs=[self.loss])
        self.pred_fn = theano.function(inputs=[self.image],outputs=[self.logits])
    
    def init_tparams(self):
        batch_size = self.hparams['mini_batch_size']
        kernel_size = self.hparams['init_conv_kernel_size']
        h = self.hparams['h']
        two_h = 2*h
        shape = [2*h,3] + kernel_size
        self.tparams['conv_7x7_W'] = glorot(shape=shape, fan_in=(kernel_size[0]*kernel_size[1]*3), fan_out=2*h, name='conv_7x7_W', target='dev0')
        self.tparams['conv_7x7_b'] = constant(shape=(2*h,), c=0., name='conv_7x7_b', target='dev0')
        
        #residual block params
        for i in xrange(self.hparams['nb_residual_block']):
            self.tparams['conv_1x1_{}_1_W'.format(str(i))] = glorot(shape=[h,2*h,1,1], fan_in=(1*1*2*h), fan_out=h, name='conv_1x1_{}_1_W'.format(str(i)), target='dev0')
            self.tparams['conv_1x1_{}_1_b'.format(str(i))] = constant(shape=(h,), c=0., name='conv_1x1_{}_1_b'.format(str(i)), target='dev0')
            self.tparams['conv_3x3_{}_W'.format(str(i))] = glorot(shape=[h,h,3,3], fan_in=(3*3*h), fan_out=h, name='conv_3x3_{}_W'.format(str(i)), target='dev0')
            self.tparams['conv_3x3_{}_b'.format(str(i))] = constant(shape=(h,), c=0., name='conv_3x3_{}_b'.format(str(i)), target='dev0')
            self.tparams['conv_1x1_{}_2_W'.format(str(i))] = glorot(shape=[2*h,h,1,1], fan_in=(1*1*h), fan_out=2*h, name='conv_1x1_{}_2_W'.format(str(i)), target='dev0')
            self.tparams['conv_1x1_{}_2_b'.format(str(i))] = constant(shape=(2*h,), c=0., name='conv_1x1_{}_2_b'.format(str(i)), target='dev0')
        
        #2 ReLU layers
        nb_relu_units = self.hparams['nb_relu_units']
        self.tparams['conv_relu_1x1_1_W'] = glorot(shape=[nb_relu_units,2*h,1,1], fan_in=(1*1*2*h), fan_out=nb_relu_units, name='conv_relu_1x1_1_W', target='dev0')
        self.tparams['conv_relu_1x1_1_b'] = constant(shape=(nb_relu_units,), c=0., name='conv_relu_1x1_1_b', target='dev0')
        self.tparams['conv_relu_1x1_2_W'] = glorot(shape=[nb_relu_units,nb_relu_units,1,1], fan_in=(1*1*nb_relu_units), fan_out=nb_relu_units, name='conv_relu_1x1_2_W', target='dev0')
        self.tparams['conv_relu_1x1_2_b'] = constant(shape=(nb_relu_units,), c=0., name='conv_relu_1x1_2_b', target='dev0')
        
        #softmax
        self.tparams['conv_logits_W'] = glorot(shape=[3,nb_relu_units,1,1], fan_in=(1*1*nb_relu_units), fan_out=3, name='conv_logits_W', target='dev0')
        self.tparams['conv_logits_b'] = constant(shape=(3,), c=0., name='conv_logits_b', target='dev0')

    def save_params(self, name):
        self.save(name)

    def load_params(self, path):
        self.load(filename=path)
        print "params restored from {}".format(path)

    def train(self, name, train, valid, mbsz=32, nb_epochs=200):
        tidx = np.arange(len(train))
        vidx = np.arange(len(valid))
        mbsz = self.hparams['mini_batch_size']
        nb_epochs = self.hparams['nb_epochs']

        for e in xrange(nb_epochs):
            train_losses = []
            validation_losses = []

            np.random.shuffle(tidx)

            for i in xrange(0, len(tidx)/5, mbsz):
                image = train[tidx[i:i+mbsz]]
                l = self.train_op(image)
                train_losses.append(l)
                print 'training...', i, np.mean(train_losses), '\r',

            np.random.shuffle(vidx)

            for j in xrange(0, len(vidx)/5, mbsz):
                image = valid[vidx[i:i+mbsz]]
                l = self.val_op(image)
                validation_losses.append(l)
                print 'validating...', j, np.mean(validation_losses), '\r',

            path = self.save_params("{}/model.pkl".format(name))
            self.sample('{}\\sample_{}.png'.format(name, e), train[tidx[:16]])
            print "epoch: {}/{}, train loss: {}, validation loss: {}, model saved: {}".format(e,
                    nb_epochs, np.mean(train_losses), np.mean(validation_losses), path)

            del train_losses
            del validation_losses

    def sample(self, name, image, n=4):
        image[:, 16:48, 16:48, :] = 0.
        for i in xrange(16, 24):
            for j in xrange(16, 24):
                pixels = self.pred_fn(image)[0]
                
                #multi scale sampling (16 pixels at the time, 8 pix apart, since 7x7 conv)
                for step_i in xrange(0,32,8):
                    for step_j in xrange(0,32,8):
                        image[:, i+step_i, j+step_j, :] = pixels[:, i+step_i, j+step_j, :]
                print 'sampling...', i, j, '\r',

        canvas = Image.new('RGB', (72*n, 72*n))
        for i in xrange(n):
            for j in xrange(n):
                im = Image.fromarray(np.cast[np.uint8](image[i*n+j, :, :, :] * 255))
                canvas.paste(im, (72*i+4, 72*j+4))
        canvas.save(name)


    def run(self):
        expname = self.experiment_name
        datahome = 'C:\\Users\\Mariane\\Documents\\ift6266\\ift6266-h17-project\\'
        train = np.load(datahome + 'images.train.npz').items()[0][1] / 255.
        valid = np.load(datahome + 'images.valid.npz').items()[0][1] / 255.
        train = train.astype('float32')
        valid = valid.astype('float32')

        self.train(datahome+"saved\\"+expname, train, valid)
        
    # -----------------------
    # --- Private Methods ---
    # -----------------------
    def _get_default_hparams(self):
        """ Returns default hyperparameters"""
        return {
            'adam_leanring_rate':   1e-5,
            'init_conv_kernel_size':[7,7],
            'h':                    64,
            'mini_batch_size':      16,
            'nb_epochs':            1,
            'nb_residual_block':    7,
            'nb_relu_units':        32
        }

    def _get_random_hparams(self):
        # TODO: Fix random hparams selection
        hparams = {}
        return hparams
