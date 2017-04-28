"""
About:  PixelCNN implementation for the IFT6266 project
Author: Mariane Maynard
Date:   April 2017
Other:  Inspired from the tensorflow implementation of Sherjil Ozair: https://github.com/sherjilozair/ift6266/blob/master/pixelcnn.py
"""

import sys, os
import numpy as np
from lib.layers import relu, clipped_relu, conv, conv_1d, max_pool_1d, sigmoid, tanh
from lib.inits import constant, glorot, orthogonal, he
from lib.updates import Adam
from PIL import Image
import pickle as cPickle
import theano
import theano.tensor as T
from . import BaseModel


class PixelCNN(BaseModel):
    def conv2d(self, inputs, activation_fn=clipped_relu, scope=None):

        W = self.tparams[scope+'_W']
        b = self.tparams[scope+'_b']

        h = conv(inputs, W, b=b, border_mode='half', subsample=(1, 1))  #half is equivalent to same for odd kernel_size
        if activation_fn:
            h = activation_fn(h)

        return h

    def residual_block(self, x, name):
        h = self.conv2d(x, scope='conv_1x1_{}_1'.format(name))
        h = self.conv2d(h, scope='conv_3x3_{}'.format(name))
        h = self.conv2d(h, scope='conv_1x1_{}_2'.format(name))
        return h + x

    def compile(self):
        image = T.tensor4('src_images', dtype='float32')       
        output = image.dimshuffle(0,3,1,2)
        input = T.set_subtensor(output[:, :, 16:48, 16:48], 0.)
        
        caption = T.tensor3('src_captions', dtype='float32')
        embededded = self.text_embedding(caption)
               
        h = self.conv2d(input, scope='conv_7x7')
        for i in range(self.hparams['nb_residual_block']):
            h = self.residual_block(h, str(i))
            
        h = T.concatenate([h, embededded.dimshuffle(0,'x',1,2)],axis=1)
        h = self.conv2d(h, scope='conv_relu_1x1_1')
        h = self.conv2d(h, scope='conv_relu_1x1_2')
        logits = self.conv2d(h, activation_fn=sigmoid, scope='conv_logits')
        
        losses = T.nnet.binary_crossentropy(logits, output)
        losses = losses[:, :, 16:48, 16:48]
        loss = losses.sum(axis=[1, 2, 3]).mean()
        self.updater = Adam(self.hparams['adam_leanring_rate'])
        params = [self.tparams[k] for k in self.tparams.keys()]
        updates = self.updater(params, loss)
        self.train_op = theano.function(inputs=[image, caption],outputs=loss,updates=updates, mode='FAST_RUN')
        self.val_op = theano.function(inputs=[image, caption],outputs=loss, mode='FAST_RUN')
        
        logits = logits.dimshuffle(0,2,3,1)
        self.pred_fn = theano.function(inputs=[image, caption],outputs=logits, mode='FAST_RUN')

    def gru(self, input):                                       # Input is (batch, h, 64)
    
        def gru_step(x_t, h_m1):
            zt = sigmoid(bz + T.dot(h_m1, Uz) + T.dot(x_t, Wz))
            rt = sigmoid(br + T.dot(h_m1, Ur) + T.dot(x_t, Wr))
            ct = tanh(bc + T.dot(rt*h_m1, Uc) + T.dot(x_t, Wc))
            ht = (1-zt)*h_m1 + zt*ct
            
            return ht                                               # (batch, h)
    
        h = self.hparams['h']
        seq = input.dimshuffle(2, 0, 1)                         # Reshaping to (64, batch, h)
        n_steps, n_batch = seq.shape[0], seq.shape[1]
        outputs_info = T.alloc(np.asarray(0., dtype='float32'), n_batch, h)
        
        bz = self.tparams['gru_past_bz']
        Uz = self.tparams['gru_past_Uz']
        Wz = self.tparams['gru_past_Wz']
                                       
        br = self.tparams['gru_past_br']
        Ur = self.tparams['gru_past_Ur']
        Wr = self.tparams['gru_past_Wr']
                                       
        bc = self.tparams['gru_past_bc']
        Uc = self.tparams['gru_past_Uc']
        Wc = self.tparams['gru_past_Wc']
        output_past, _ = theano.scan(fn=gru_step, sequences=seq, outputs_info=outputs_info, n_steps=n_steps)
        
        bz = self.tparams['gru_future_bz']
        Uz = self.tparams['gru_future_Uz']
        Wz = self.tparams['gru_future_Wz']
    
        br = self.tparams['gru_future_br']
        Ur = self.tparams['gru_future_Ur']
        Wr = self.tparams['gru_future_Wr']
    
        bc = self.tparams['gru_future_bc']
        Uc = self.tparams['gru_future_Uc']
        Wc = self.tparams['gru_future_Wc']
        output_future, _ = theano.scan(fn=gru_step, sequences=seq[::-1,:,:], outputs_info=outputs_info, n_steps=n_steps)
        # output is (nsteps, batch, h)

        output = T.concatenate([output_past, output_future[::-1,:,:]], axis=2)
        output = output.dimshuffle(1, 2, 0)                     # Reshaping back to (batch, 2*h, 64)
        return tanh(output)
    
    def text_embedding(self, input):
        a1 = conv_1d(input, self.tparams['char_conv_1_W'], b=self.tparams['char_conv_1_b'], border_mode=(2,), subsample=(1,))   #same (bsz, 2h, 256)
        h1 = clipped_relu(a1, max=16.)
        
        a2 = conv_1d(h1, self.tparams['char_conv_2_W'], b=self.tparams['char_conv_2_b'], border_mode=(1,), subsample=(2,))      #halving the output (bsz, 2h, 128)
        h2 = clipped_relu(a2, max=16.)
        
        a3 = conv_1d(h2, self.tparams['char_conv_3_W'], b=self.tparams['char_conv_3_b'], border_mode=(2,), subsample=(1,))      #same (bsz, h, 128)
        h3 = clipped_relu(a3, max=16.)
        
        a4 = conv_1d(h3, self.tparams['char_conv_4_W'], b=self.tparams['char_conv_4_b'], border_mode=(1,), subsample=(2,))      #halving the output (bsz, h, 64)
        h4 = clipped_relu(a4, max=16.)
        
        hgru = self.gru(h4)         #(bsz, 2h, 64)
        
        output = T.dot(hgru.dimshuffle(0,2,1), self.tparams['char_concat_W']) + self.tparams['char_concat_b']                   #(bsz, 64, 1)
        output = T.dot(output.dimshuffle(0,2,1), self.tparams['char_attention_W']) + self.tparams['char_attention_b']           #(bsz, 1, 64)
        embededded_2d = T.batched_dot(output.dimshuffle(0,2,1),output)                                                          #outer product : (bsz, 64, 64)
        return sigmoid(embededded_2d)
    
    def one_hot_alphabet_encoding(self, captions):
        alphabet = self.hparams['alphabet']
        maxlen = self.hparams['caption_max_length']
        encodded_captions = np.zeros(shape=(len(captions),len(alphabet),maxlen), dtype='float32')  #batch_size, alphabet_size, maxlen
        
        for i, capx in enumerate(captions):
            #since there are 5 or more, choose a caption randomly
            cap = capx[np.random.randint(0, len(capx))].lower()
            for j, char in enumerate(cap):
                ix = self.char_to_ix[char]
                encodded_captions[i,ix,j] = 1.
                
        return encodded_captions
    
    def build(self):
        self.init_tparams()
    
    def init_tparams(self):
        batch_size = self.hparams['mini_batch_size']
        kernel_size = self.hparams['init_conv_kernel_size']
        h = self.hparams['h']
        shape = [2*h,3] + kernel_size
        self.tparams['conv_7x7_W'] = glorot(shape=shape, fan_in=(kernel_size[0]*kernel_size[1]*3), fan_out=(2*h), name='conv_7x7_W', target='dev0')
        self.tparams['conv_7x7_b'] = constant(shape=(2*h), c=0., name='conv_7x7_b', target='dev0')
        
        #residual block params
        for i in range(self.hparams['nb_residual_block']):
            self.tparams['conv_1x1_{}_1_W'.format(str(i))] = glorot(shape=[h,2*h,1,1], fan_in=(1*1*2*h), fan_out=h, name='conv_1x1_{}_1_W'.format(str(i)), target='dev0')
            self.tparams['conv_1x1_{}_1_b'.format(str(i))] = constant(shape=(h), c=0., name='conv_1x1_{}_1_b'.format(str(i)), target='dev0')
            self.tparams['conv_3x3_{}_W'.format(str(i))] = glorot(shape=[h,h,3,3], fan_in=(3*3*h), fan_out=h, name='conv_3x3_{}_W'.format(str(i)), target='dev0')
            self.tparams['conv_3x3_{}_b'.format(str(i))] = constant(shape=(h), c=0., name='conv_3x3_{}_b'.format(str(i)), target='dev0')
            self.tparams['conv_1x1_{}_2_W'.format(str(i))] = glorot(shape=[2*h,h,1,1], fan_in=(1*1*h), fan_out=2*h, name='conv_1x1_{}_2_W'.format(str(i)), target='dev0')
            self.tparams['conv_1x1_{}_2_b'.format(str(i))] = constant(shape=(2*h), c=0., name='conv_1x1_{}_2_b'.format(str(i)), target='dev0')
        
        #2 ReLU layers
        nb_relu_units = self.hparams['nb_relu_units']
        self.tparams['conv_relu_1x1_1_W'] = glorot(shape=[nb_relu_units,2*h+1,1,1], fan_in=(1*1*2*h+1), fan_out=nb_relu_units, name='conv_relu_1x1_1_W', target='dev0')
        self.tparams['conv_relu_1x1_1_b'] = constant(shape=(nb_relu_units), c=0., name='conv_relu_1x1_1_b', target='dev0')
        self.tparams['conv_relu_1x1_2_W'] = glorot(shape=[nb_relu_units,nb_relu_units,1,1], fan_in=(1*1*nb_relu_units), fan_out=nb_relu_units, name='conv_relu_1x1_2_W', target='dev0')
        self.tparams['conv_relu_1x1_2_b'] = constant(shape=(nb_relu_units), c=0., name='conv_relu_1x1_2_b', target='dev0')
        
        #softmax
        self.tparams['conv_logits_W'] = glorot(shape=[3,nb_relu_units,1,1], fan_in=(1*1*nb_relu_units), fan_out=3, name='conv_logits_W', target='dev0')
        self.tparams['conv_logits_b'] = constant(shape=(3), c=0., name='conv_logits_b', target='dev0')
        
        #Char-CNN-...
        
        alphabet = self.hparams['alphabet']
        alphabet_size = len(alphabet)
        self.char_to_ix = {ch: ix for ix, ch in enumerate(alphabet)}
        self.tparams['char_conv_1_W'] = glorot(shape=[2*h, alphabet_size, 5], fan_in=(5*alphabet_size), fan_out=2*h, name='char_conv_1_W', target='dev0')
        self.tparams['char_conv_1_b'] = constant(shape=(2*h), c=0., name='char_conv_1_b', target='dev0')
        self.tparams['char_conv_2_W'] = glorot(shape=[2*h,2*h,4], fan_in=(4*2*h), fan_out=2*h, name='char_conv_2_W', target='dev0')
        self.tparams['char_conv_2_b'] = constant(shape=(2*h), c=0., name='char_conv_2_b', target='dev0')
        self.tparams['char_conv_3_W'] = glorot(shape=[h, 2*h, 5], fan_in=(5*2*h), fan_out=h, name='char_conv_3_W', target='dev0')
        self.tparams['char_conv_3_b'] = constant(shape=(h), c=0., name='char_conv_3_b', target='dev0')
        self.tparams['char_conv_4_W'] = glorot(shape=[h,h,4], fan_in=(4*h), fan_out=h, name='char_conv_4_W', target='dev0')
        self.tparams['char_conv_4_b'] = constant(shape=(h), c=0., name='char_conv_4_b', target='dev0')
        
        #...-GRU
        self.tparams['gru_past_bz'] = constant(shape=h, c=0., name='gru_past_bz', target='dev0')
        self.tparams['gru_past_Uz'] = orthogonal(shape=(h,h), scale=1., name='gru_past_Uz', target='dev0')
        self.tparams['gru_past_Wz'] = orthogonal(shape=(h,h), scale=1., name='gru_past_Wz', target='dev0')

        self.tparams['gru_past_br'] = constant(shape=h, c=0., name='gru_past_br', target='dev0')
        self.tparams['gru_past_Ur'] = orthogonal(shape=(h,h), scale=1., name='gru_past_Ur', target='dev0')
        self.tparams['gru_past_Wr'] = orthogonal(shape=(h,h), scale=1., name='gru_past_Wr', target='dev0')

        self.tparams['gru_past_bc'] = constant(shape=h, c=0., name='gru_past_bc', target='dev0')
        self.tparams['gru_past_Uc'] = orthogonal(shape=(h,h), scale=1., name='gru_past_Uc', target='dev0')
        self.tparams['gru_past_Wc'] = orthogonal(shape=(h,h), scale=1., name='gru_past_Wc', target='dev0')
        
        self.tparams['gru_future_bz'] = constant(shape=h, c=0., name='gru_future_bz', target='dev0')
        self.tparams['gru_future_Uz'] = orthogonal(shape=(h,h), scale=1., name='gru_future_Uz', target='dev0')
        self.tparams['gru_future_Wz'] = orthogonal(shape=(h,h), scale=1., name='gru_future_Wz', target='dev0')

        self.tparams['gru_future_br'] = constant(shape=h, c=0., name='gru_future_br', target='dev0')
        self.tparams['gru_future_Ur'] = orthogonal(shape=(h,h), scale=1., name='gru_future_Ur', target='dev0')
        self.tparams['gru_future_Wr'] = orthogonal(shape=(h,h), scale=1., name='gru_future_Wr', target='dev0')

        self.tparams['gru_future_bc'] = constant(shape=h, c=0., name='gru_future_bc', target='dev0')
        self.tparams['gru_future_Uc'] = orthogonal(shape=(h,h), scale=1., name='gru_future_Uc', target='dev0')
        self.tparams['gru_future_Wc'] = orthogonal(shape=(h,h), scale=1., name='gru_future_Wc', target='dev0')
        
        self.tparams['char_concat_W'] = glorot(shape=[2*h,1], fan_in=(2*h), fan_out=1, name='char_concat_W', target='dev0')
        self.tparams['char_concat_b'] = constant(shape=(1), c=0., name='char_concat_b', target='dev0')
        self.tparams['char_attention_W'] = glorot(shape=[64,64], fan_in=(64), fan_out=64, name='char_attention_W', target='dev0')
        self.tparams['char_attention_b'] = constant(shape=(64), c=0., name='char_attention_b', target='dev0')
        

    def save_params(self, name):
        self.save(name)
        return name

    def load_params(self, path):
        self.load(filename=path)
        str = "params restored from {}".format(path)
        print(str)

    def train(self, name, train, valid, captions):
        mbsz = self.hparams['mini_batch_size']
        nb_epochs = self.hparams['nb_epochs']
        training_set_fraction = self.hparams['training_set_fraction']
        
        train_images = train['values'].astype('float32') / 255.
        train_caption_ids = train['keys']
        train_captions = self.one_hot_alphabet_encoding([captions[k] for k in train_caption_ids])
        tidx = np.arange(len(train_images))
        
        valid_images = valid['values'].astype('float32') / 255.
        valid_caption_ids = valid['keys']
        valid_captions = self.one_hot_alphabet_encoding([captions[k] for k in valid_caption_ids])
        vidx = np.arange(len(valid_images))

        for e in range(nb_epochs):
            train_losses_mean = 0
            valid_losses_mean = 0
            train_losses = 0
            validation_losses = 0

            np.random.shuffle(tidx)

            for i in range(0, len(tidx)//training_set_fraction, mbsz):
                image = train_images[tidx[i:i+mbsz]]
                caption = train_captions[tidx[i:i+mbsz]]
                l = self.train_op(image, caption)
                train_losses += l
                train_losses_mean = train_losses*mbsz/(i+mbsz)
                print 'training...', i, train_losses_mean, '\r', #mean

            np.random.shuffle(vidx)

            for j in range(0, len(vidx)//training_set_fraction, mbsz):
                image = valid_images[vidx[j:j+mbsz]]
                caption = valid_captions[vidx[j:j+mbsz]]
                l = self.val_op(image, caption)
                validation_losses += l
                valid_losses_mean = validation_losses*mbsz/(j+mbsz)
                print 'validating...', j, valid_losses_mean, '\r',

            path = self.save_params("{}\\model.pkl".format(name))
            self.sample('{}\\sample_{}.png'.format(name, e), valid_images[vidx[:16]], valid_captions[vidx[:16]])
            print("epoch: {}/{}, train loss: {}, validation loss: {}, model saved: {}".format(e+1,
                    nb_epochs, train_losses_mean, valid_losses_mean, path))

    def sample(self, name, image, caption, n=4):
        image[:, 16:48, 16:48, :] = 0.
        pixels = self.pred_fn(image, caption)
        image[:, 16:48, 16:48, :] = pixels[:, 16:48, 16:48, :]

        canvas = Image.new('RGB', (72*n, 72*n))
        for i in range(n):
            for j in range(n):
                im = Image.fromarray(np.cast[np.uint8](image[i*n+j, :, :, :]* 255))
                canvas.paste(im, (72*i+4, 72*j+4))
        canvas.save(name)


    def run(self):
        expname = self.experiment_name
        datahome = 'C:\\Users\\Mariane\\Documents\\ift6266\\ift6266-h17-project\\'
        train = np.load(datahome + 'images.train.npz').items()[0][1] 
        valid = np.load(datahome + 'images.valid.npz').items()[0][1] 
        with open(datahome+'dict_key_imgID_value_caps_train_and_valid.pkl', 'rb') as f:
            captions = cPickle.load(f)

        self.compile()
        self.train(datahome+"saved\\"+expname, train, valid, captions)
        
    # -----------------------
    # --- Private Methods ---
    # -----------------------
    def _get_default_hparams(self):
        """ Returns default hyperparameters"""
        return {
            'adam_leanring_rate':   1e-3,
            'alphabet':             'abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:\'"/\\|_@#$%^&*~`+-=<>()[]{} \n',
            'caption_max_length':   256,        #actually 250
            'init_conv_kernel_size':[7,7],
            'h':                    32,
            'mini_batch_size':      16,
            'nb_epochs':            1,
            'nb_residual_block':    15,
            'nb_relu_units':        32,
			'training_set_fraction':32
        }

    def _get_random_hparams(self):
        # TODO: Fix random hparams selection
        hparams = {}
        return self._get_default_hparams()
