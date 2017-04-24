"""
About:  PixelCNN implementation for the IFT6266 project
Author: Mariane Maynard
Date:   April 2017
Other:  Inspired from the tensorflow implementation of Sherjil Ozair: https://github.com/sherjilozair/ift6266/blob/master/pixelcnn.py
"""

import sys, os
import numpy as np
from lib.layers import relu, clipped_relu, conv, conv_1d, sigmoid, deconv, tanh, batchnorm, concat, max_pool, max_pool_1d, avg_pool, dropout, BinaryCrossEntropy, MeanSquaredError, MeanAbsoluteError
from lib.inits import constant, he, normal, orthogonal, uniform, glorot
from lib.utils import castX, shared0s, sharedX, floatX
from lib.updates import Adam, Regularizer
from PIL import Image
import theano
import theano.tensor as T
from . import BaseModel

class nn:

    #Using 5 dimensioned tensors as input, 6-dim as W and a vector for b
    #inputs : (batch_size, nb_of_features, nb_of_color_channels, height, width)
    #W : (nb_of_output_features, nb_of_input_features, nb_of_input_color_channels, nb_of_output_color_channels, height, width)
    #b : (nb_of_output_features, nb_of_output_color_channels)
    @staticmethod
    def conv2d(inputs, num_outputs, kernel_size, params, mask=None, activation_fn=clipped_relu, scope=None, input_shape=None):
        input_shape = inputs.shape
        #shape = [num_outputs, input_shape[1]] + [3,3] + kernel_size
        nb_of_color_channels = 3

        W = params[scope+'_W']
        b = params[scope+'_b']

        if mask:
            #mid_x = shape[3]/2
            #mid_y = shape[4]/2

            #mask_filter = T.ones(shape, dtype='float32')
            #mask_filter = T.set_subtensor(mask_filter[:,:,:,:,mid_x, mid_y+1:], 0.)
            #mask_filter = T.set_subtensor(mask_filter[:,:,:,:,mid_x+1:, :], 0.)
            
            #for each color
            #for i in range(nb_of_color_channels):
            #    mask_filter = T.set_subtensor(mask_filter[:,:,i+1:,i,mid_x, mid_y], 0.)

            #    if mask == 'a':
            #        mask_filter = T.set_subtensor(mask_filter[:,:,i,i,mid_x, mid_y], 0.)
                
            W  = W * mask #mask_filter

        border_mode = (kernel_size[0]//2,kernel_size[1]//2)
        subsample = (1,1)
        
        h = T.zeros(shape=[input_shape[0], num_outputs,3,64,64])
        for i in range(nb_of_color_channels):
            for j in range(nb_of_color_channels):
                result = h[:,:,i,:,:] + T.nnet.conv2d(input=inputs[:,:,j,:,:], filters=W[:,:,j,i,:,:], border_mode=border_mode, subsample=subsample, filter_flip=True)
                h = T.set_subtensor(h[:,:,i,:,:], result)
        h += b.dimshuffle('x', 0, 1, 'x', 'x')

        if activation_fn:
            h = activation_fn(h)

        return h

    @staticmethod
    def residual_block(x, name, params, masks=None, h_size=0):
        h = nn.conv2d(x, h_size, [1, 1], params, mask=masks[0], scope='conv_1x1_{}_1'.format(name))
        h = nn.conv2d(h, h_size, [3, 3], params, mask=masks[1], scope='conv_3x3_{}'.format(name))
        h = nn.conv2d(h, 2*h_size, [1, 1], params, mask=masks[2], scope='conv_1x1_{}_2'.format(name))
        return h + x


#def xrange(start, stop, step):
#    return range(start, stop, step)

#def xrange(stop):
#    return range(stop)

class PixelCNN(BaseModel):
    def compile(self):
        self.image = T.tensor4('src_images', dtype='float32')
        input = self.image.dimshuffle(0,'x',3,1,2)
        nb_h = self.hparams['h']
        nb_relu_units = self.hparams['nb_relu_units']
        h = nn.conv2d(input, 2*nb_h, [7, 7], self.tparams, mask=self.masks['mask_7x7_a'], scope='conv_7x7')
        
        for i in range(self.hparams['nb_residual_block']):
            h = nn.residual_block(h, str(i), self.tparams, masks=(self.masks['mask_1x1_1_b'],self.masks['mask_3x3_b'], self.masks['mask_1x1_2_b']), h_size=nb_h)
        h1 = nn.conv2d(h, nb_relu_units, [1, 1], self.tparams, mask=self.masks['mask_relu_1x1_1_b'], scope='conv_relu_1x1_1')
        h2 = nn.conv2d(h1, nb_relu_units, [1, 1], self.tparams, mask=self.masks['mask_relu_1x1_2_b'], scope='conv_relu_1x1_2')
        self.logits = nn.conv2d(h2, 1, [1, 1], self.tparams, scope='conv_logits')
        
        self.losses = T.sqr(self.logits - input)
        self.logits = self.logits[:,0,:,:,:].dimshuffle(0,2,3,1)
        self.losses = self.losses[:, :, :, 16:48, 16:48]
        self.loss = self.losses.sum(axis=[1, 2, 3, 4]).mean()
        self.updater = Adam(self.hparams['adam_leanring_rate'])
        self.params = [self.tparams[k] for k in self.tparams.keys()]
        self.updates = self.updater(self.params, self.loss)
        self.train_op = theano.function(inputs=[self.image],outputs=self.loss,updates=self.updates)
        self.val_op = theano.function(inputs=[self.image],outputs=self.loss)
        self.pred_fn = theano.function(inputs=[self.image],outputs=self.logits)
    
    def build(self):
        self.init_tparams()
        self.build_masks()
    
    def build_mask(self, shape, mask_type='b', name=None, target='dev0'):
        mid_x = shape[4]//2
        mid_y = shape[5]//2
        nb_of_color_channels = shape[2]

        mask_filter = constant(shape=shape, c=1., name=name, target=target)
        if(mid_y != 0):
            mask_filter = T.set_subtensor(mask_filter[:,:,:,:,mid_x, mid_y+1:], 0.)
        if(mid_x != 0):
            mask_filter = T.set_subtensor(mask_filter[:,:,:,:,mid_x+1:, :], 0.)
        
        #for each color
        for i in range(nb_of_color_channels):
            if(i < (nb_of_color_channels-1)):
                mask_filter = T.set_subtensor(mask_filter[:,:,i+1:,i,mid_x, mid_y], 0.)

            if mask_type == 'a':
                mask_filter = T.set_subtensor(mask_filter[:,:,i,i,mid_x, mid_y], 0.)
        return mask_filter
    
    def build_masks(self):
        self.masks = {}
        kernel_size = self.hparams['init_conv_kernel_size']
        h = self.hparams['h']
        nb_relu_units = self.hparams['nb_relu_units']
        self.masks['mask_7x7_a'] = self.build_mask(shape=[2*h,1,3,3] + kernel_size, mask_type='a', name='mask_7x7_a', target='dev0')
        self.masks['mask_3x3_b'] = self.build_mask(shape=[h,h,3,3,3,3], mask_type='b', name='mask_3x3_b', target='dev0')
        self.masks['mask_1x1_1_b'] = self.build_mask(shape=[h,2*h,3,3,1,1], mask_type='b', name='mask_1x1_1_b', target='dev0')
        self.masks['mask_1x1_2_b'] = self.build_mask(shape=[2*h,h,3,3,1,1], mask_type='b', name='mask_1x1_2_b', target='dev0')
        self.masks['mask_relu_1x1_1_b'] = self.build_mask(shape=[nb_relu_units,2*h,3,3,1,1], mask_type='b', name='mask_relu_1x1_1_b', target='dev0')
        self.masks['mask_relu_1x1_2_b'] = self.build_mask(shape=[nb_relu_units,nb_relu_units,3,3,1,1], mask_type='b', name='mask_relu_1x1_2_b', target='dev0')
    
    def init_tparams(self):
        batch_size = self.hparams['mini_batch_size']
        kernel_size = self.hparams['init_conv_kernel_size']
        h = self.hparams['h']
        two_h = 2*h
        shape = [2*h,1,3,3] + kernel_size
        self.tparams['conv_7x7_W'] = glorot(shape=shape, fan_in=(kernel_size[0]*kernel_size[1]*3), fan_out=(2*h*3), name='conv_7x7_W', target='dev0')
        self.tparams['conv_7x7_b'] = constant(shape=(2*h,3), c=0., name='conv_7x7_b', target='dev0')
        
        #residual block params
        for i in range(self.hparams['nb_residual_block']):
            self.tparams['conv_1x1_{}_1_W'.format(str(i))] = glorot(shape=[h,2*h,3,3,1,1], fan_in=(1*1*2*h*3), fan_out=h*3, name='conv_1x1_{}_1_W'.format(str(i)), target='dev0')
            self.tparams['conv_1x1_{}_1_b'.format(str(i))] = constant(shape=(h,3), c=0., name='conv_1x1_{}_1_b'.format(str(i)), target='dev0')
            self.tparams['conv_3x3_{}_W'.format(str(i))] = glorot(shape=[h,h,3,3,3,3], fan_in=(3*3*h*3), fan_out=h*3, name='conv_3x3_{}_W'.format(str(i)), target='dev0')
            self.tparams['conv_3x3_{}_b'.format(str(i))] = constant(shape=(h,3), c=0., name='conv_3x3_{}_b'.format(str(i)), target='dev0')
            self.tparams['conv_1x1_{}_2_W'.format(str(i))] = glorot(shape=[2*h,h,3,3,1,1], fan_in=(1*1*h*3), fan_out=2*h*3, name='conv_1x1_{}_2_W'.format(str(i)), target='dev0')
            self.tparams['conv_1x1_{}_2_b'.format(str(i))] = constant(shape=(2*h,3), c=0., name='conv_1x1_{}_2_b'.format(str(i)), target='dev0')
        
        #2 ReLU layers
        nb_relu_units = self.hparams['nb_relu_units']
        self.tparams['conv_relu_1x1_1_W'] = glorot(shape=[nb_relu_units,2*h,3,3,1,1], fan_in=(1*1*2*h*3), fan_out=nb_relu_units*3, name='conv_relu_1x1_1_W', target='dev0')
        self.tparams['conv_relu_1x1_1_b'] = constant(shape=(nb_relu_units,3), c=0., name='conv_relu_1x1_1_b', target='dev0')
        self.tparams['conv_relu_1x1_2_W'] = glorot(shape=[nb_relu_units,nb_relu_units,3,3,1,1], fan_in=(1*1*nb_relu_units*3), fan_out=nb_relu_units*3, name='conv_relu_1x1_2_W', target='dev0')
        self.tparams['conv_relu_1x1_2_b'] = constant(shape=(nb_relu_units,3), c=0., name='conv_relu_1x1_2_b', target='dev0')
        
        #softmax
        self.tparams['conv_logits_W'] = glorot(shape=[1,nb_relu_units,3,3,1,1], fan_in=(1*1*nb_relu_units*3), fan_out=3, name='conv_logits_W', target='dev0')
        self.tparams['conv_logits_b'] = constant(shape=(1,3), c=0., name='conv_logits_b', target='dev0')

    def save_params(self, name):
        self.save(name)
        return name

    def load_params(self, path):
        self.load(filename=path)
        str = "params restored from {}".format(path)
        print(str)

    def train(self, name, train, valid, mbsz=32, nb_epochs=200):
        tidx = np.arange(len(train))
        vidx = np.arange(len(valid))
        mbsz = self.hparams['mini_batch_size']
        nb_epochs = self.hparams['nb_epochs']
        training_set_fraction = self.hparams['training_set_fraction']

        for e in range(nb_epochs):
            train_losses = []
            validation_losses = []

            np.random.shuffle(tidx)

            for i in range(0, len(tidx)//training_set_fraction, mbsz):
                image = train[tidx[i:i+mbsz]]
                l = self.train_op(image)
                train_losses.append(l)
                print('training...', i, np.mean(train_losses), end='\r')

            np.random.shuffle(vidx)

            for j in range(0, len(vidx), mbsz):
                image = valid[vidx[j:j+mbsz]]
                l = self.val_op(image)
                validation_losses.append(l)
                print('validating...', j, np.mean(validation_losses), end='\r')

            path = self.save_params("{}\\model.pkl".format(name))
            self.sample('{}\\sample_{}.png'.format(name, e), train[tidx[:16]])
            print("epoch: {}/{}, train loss: {}, validation loss: {}, model saved: {}".format(e,
                    nb_epochs, np.mean(train_losses), np.mean(validation_losses), path))

            del train_losses
            del validation_losses

    def sample(self, name, image, n=4):
        image[:, 16:48, 16:48, :] = 0.
        for i in range(16, 48):
            for j in range(16, 48):
                pixels = self.pred_fn(image)
                image[:, i, j, :] = pixels[:, i, j, :]
                #multi scale sampling (16 pixels at the time, 8 pix apart, since 7x7 conv)
                #for step_i in xrange(0,32,8):
                #    for step_j in xrange(0,32,8):
                #        image[:, i+step_i, j+step_j, :] = pixels[:, i+step_i, j+step_j, :]
                print('sampling...', i, j, end='\r')

        canvas = Image.new('RGB', (72*n, 72*n))
        for i in range(n):
            for j in range(n):
                im = Image.fromarray(np.cast[np.uint8](image[i*n+j, :, :, :])) #* 255))
                canvas.paste(im, (72*i+4, 72*j+4))
        canvas.save(name)


    def run(self):
        expname = self.experiment_name
        datahome = 'C:\\Users\\Mariane\\Documents\\ift6266-h17-project\\'
        train = np.load(datahome + 'images.train.npz').items()[0][1] #/ 255.
        valid = np.load(datahome + 'images.valid.npz').items()[0][1] #/ 255.
        train = train.astype('float32')
        valid = valid.astype('float32')

        self.compile()
        self.train(datahome+"saved\\"+expname, train, valid)
        
    # -----------------------
    # --- Private Methods ---
    # -----------------------
    def _get_default_hparams(self):
        """ Returns default hyperparameters"""
        return {
            'adam_leanring_rate':   1e-1,
            'init_conv_kernel_size':[7,7],
            'h':                    32,
            'mini_batch_size':      16,
            'nb_epochs':            2,
            'nb_residual_block':    4,
            'nb_relu_units':        32,
			'training_set_fraction':8
        }

    def _get_random_hparams(self):
        # TODO: Fix random hparams selection
        hparams = {}
        return hparams
