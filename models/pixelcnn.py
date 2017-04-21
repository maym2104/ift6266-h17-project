import sys, os
import numpy as np
#import tensorflow as tf
#import tensorflow.contrib.slim as slim
from lib.layers import lrelu, relu, conv, conv_1d, sigmoid, deconv, tanh, batchnorm, concat, max_pool, max_pool_1d, avg_pool, dropout, BinaryCrossEntropy, MeanSquaredError, MeanAbsoluteError
from lib.inits import constant, he, normal, orthogonal, uniform
from lib.utils import castX, shared0s, sharedX, floatX
from lib.updates import Adam, Regularizer
from PIL import Image
import theano
import theano.tensor as T
from . import BaseModel

class nn:

    @staticmethod
    def conv2d(inputs, num_outputs, kernel_size, params, mask=None, activation_fn=relu, scope=None, input_shape=None):
        #ishape = input_shape if input_shape is not None else inputs.shape.eval()
        shape = [num_outputs, inputs.shape[1]] + kernel_size
        #weights_initializer = tf.contrib.layers.xavier_initializer()

        #with tf.variable_scope(scope):
            #W = tf.get_variable('W', shape, tf.float32, weights_initializer)
        #W = he(shape=shape, fan_in=(kernel_size[0]*kernel_size[1]*num_outputs), name=scope+'W', target='dev0')
        W = params[scope+'_W']    #T.tensor4(scope+'_W', dtype='float32')
            #b = tf.get_variable('b', num_outputs, tf.float32, tf.zeros_initializer())
        #b = constant(shape=(num_outputs), c=0., name=scope+'b', target='dev0')
        #b = T.zeros(shape=num_outputs, dtype='float32')
        b = params[scope+'_b']  #T.vector(scope+'_b', dtype='float32')

        if mask:
            mid_x = shape[2]/2
            mid_y = shape[3]/2
            mask_filter = T.ones(shape, dtype='float32')
            mask_filter = T.set_subtensor(mask_filter[:,:,mid_x, mid_y+1:], 0.)
            #mask_filter[mid_x, mid_y+1:, :, :] = False
            mask_filter = T.set_subtensor(mask_filter[:,:,mid_x+1:, :], 0.)
            #mask_filter[mid_x+1:, :, :, :] = False

            if mask == 'a':
                #mask_filter[mid_x, mid_y, :, :] = False
                mask_filter = T.set_subtensor(mask_filter[:,:,mid_x, mid_y], 0.)
                
            #mask_filter = T.tensor4(mask, dtype='float32')
            W  = W * mask_filter

        #h = tf.add(tf.nn.conv2d(inputs, W, strides=[1,1,1,1], padding='SAME'), b)
        h = conv(inputs, W, b, border_mode=(kernel_size[0]/2,kernel_size[1]/2), subsample=(1,1))

        if activation_fn:
            h = activation_fn(h)

        return h

    @staticmethod
    def residual_block(x, name, params, mask=None, h_size=0):
        #nr_filters = x.shape[-1] #h_size if h_size is not 0 else x.shape.eval()[-1]
        h = nn.conv2d(x, h_size, [1, 1], params, mask=mask, scope='conv_1x1_{}_1'.format(name))
        h = nn.conv2d(h, h_size, [3, 3], params, mask=mask, scope='conv_3x3_{}'.format(name))
        h = nn.conv2d(h, 2*h_size, [1, 1], params, mask=mask, scope='conv_1x1_{}_2'.format(name))
        return h + x


class PixelCNN(BaseModel):
    #mask = np.ones((64, 64, 1), dtype=np.bool)
    #mask[16:48, 16:48, :] = False

    def build(self):
        #self.image = uniform((32,28,28,1), scale=0.05, name='image', shared=True, target=None)
        #self.image = tf.placeholder(tf.float32, [None, 28, 28, 1])
        self.image = T.tensor4('src_images', dtype='float32')
        input = self.image.dimshuffle(0,3,1,2)
        #def initNet(input):
        self.init_tparams()
        h = nn.conv2d(input, 256, [7, 7], self.tparams, mask='a', scope='conv_7x7')
        #conv7x7 = theano.function([self.image], nn.conv2d(self.image, 256, [7, 7], mask='a', scope='conv_7x7', input_shape=(None, 28, 28, 1)), name='conv_7x7')
        for i in xrange(self.hparams['nb_residual_block']):
            h = nn.residual_block(h, str(i), self.tparams, mask='b', h_size=self.hparams['h'])
        h1 = nn.conv2d(h, 32, [1, 1], self.tparams, mask='b', scope='conv_relu_1x1_1')
        h2 = nn.conv2d(h1, 256, [1, 1], self.tparams, mask='b', scope='conv_relu_1x1_2')
        self.logits = nn.conv2d(h2, 3, [1, 1], self.tparams, activation_fn=sigmoid, scope='conv_logits')

        #self.preds = sigmoid(self.logits)
        #self.losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=self.image)
        self.losses = T.nnet.binary_crossentropy(self.logits, input)
		
        #mask = T.zeros(shape=self.losses.shape, dtype='uint8')
        #mask = T.set_subtensor(mask[:, 16:48, 16:48, :], 1)
        #self.losses = mask * self.losses 
        self.losses = self.losses[:, :, 16:48, 16:48]
        self.loss = self.losses.sum(axis=[1, 2, 3]).mean()  #T.mean(T.sum(self.losses, axis=[1, 2, 3]))
        #self.train_op = tf.train.AdamOptimizer(1e-5).minimize(self.loss)
        self.updater = Adam(self.hparams['adam_leanring_rate'])
        self.params = [self.tparams[k] for k in self.tparams.keys()]
        self.updates = self.updater(self.params, self.loss)
        self.train_op = theano.function(inputs=[self.image],outputs=[self.loss],updates=self.updates)
        self.pred_fn = theano.function(inputs=[self.image],outputs=[self.logits])
        #self.train_op = self.updater(self.params, self.loss)
        #self.sess = tf.Session()
        #self.sess.run(tf.global_variables_initializer())
        #f = theano.function([self.image], initNet(self.image))
        #self.saver = tf.train.Saver()
    
    def mask_builder(self, shape, mask='b', name=None, target=None):
        mid_x = shape[0]/2
        mid_y = shape[1]/2
        mask_filter = constant(shape=shape, c=1., name=name, target='dev0') #T.ones(shape, dtype='float32')
        mask_filter = T.set_subtensor(mask_filter[mid_x, mid_y+1:, :, :], 0.)
        #mask_filter[mid_x, mid_y+1:, :, :].set_value(0.)
        mask_filter = T.set_subtensor(mask_filter[mid_x+1:, :, :, :], 0.)
        #mask_filter[mid_x+1:, :, :, :].set_value(0.)

        if mask == 'a':
            #mask_filter[mid_x, mid_y, :, :].set_value(0.)
            mask_filter = T.set_subtensor(mask_filter[mid_x, mid_y, :, :], 0.)
            
        return mask_filter
    
    def init_tparams(self):
        batch_size = self.hparams['mini_batch_size']
        #self.gpu_dataset = shared0s(shape=(batch_size, 1, 28, 28), dtype='float32', name='src_images', target='dev0')
        #self.mask = {}
    
        kernel_size = self.hparams['init_conv_kernel_size']
        h = self.hparams['h']
        two_h = 2*h
        shape = [two_h,3] + kernel_size
        self.tparams['conv_7x7_W'] = he(shape=shape, fan_in=(kernel_size[0]*kernel_size[1]*3), name='conv_7x7_W', target='dev0')
        self.tparams['conv_7x7_b'] = constant(shape=(two_h,), c=0., name='conv_7x7_b', target='dev0')
        #self.mask['a'] = self.mask_builder(shape, mask='a', name='a', target='dev0')
        
        #residual block params
        #self.mask['b'] = self.mask_builder([h/2,h/2,3,3], mask='b', name='b', target='dev0')
        for i in xrange(self.hparams['nb_residual_block']):
            self.tparams['conv_1x1_{}_1_W'.format(str(i))] = he(shape=[h,two_h,1,1], fan_in=(1*1*two_h), name='conv_1x1_{}_1_W'.format(str(i)), target='dev0')
            self.tparams['conv_1x1_{}_1_b'.format(str(i))] = constant(shape=(h,), c=0., name='conv_1x1_{}_1_b'.format(str(i)), target='dev0')
            self.tparams['conv_3x3_{}_W'.format(str(i))] = he(shape=[h,h,3,3], fan_in=(3*3*h), name='conv_3x3_{}_W'.format(str(i)), target='dev0')
            self.tparams['conv_3x3_{}_b'.format(str(i))] = constant(shape=(h,), c=0., name='conv_3x3_{}_b'.format(str(i)), target='dev0')
            self.tparams['conv_1x1_{}_2_W'.format(str(i))] = he(shape=[two_h,h,1,1], fan_in=(1*1*h), name='conv_1x1_{}_2_W'.format(str(i)), target='dev0')
            self.tparams['conv_1x1_{}_2_b'.format(str(i))] = constant(shape=(two_h,), c=0., name='conv_1x1_{}_2_b'.format(str(i)), target='dev0')
        
        #2 ReLU layers
        nb_relu_units = self.hparams['nb_relu_units']
        self.tparams['conv_relu_1x1_1_W'] = he(shape=[nb_relu_units,two_h,1,1], fan_in=(1*1*two_h), name='conv_relu_1x1_1_W', target='dev0')
        self.tparams['conv_relu_1x1_1_b'] = constant(shape=(nb_relu_units,), c=0., name='conv_relu_1x1_1_b', target='dev0')
        self.tparams['conv_relu_1x1_2_W'] = he(shape=[two_h,nb_relu_units,1,1], fan_in=(1*1*nb_relu_units), name='conv_relu_1x1_2_W', target='dev0')
        self.tparams['conv_relu_1x1_2_b'] = constant(shape=(two_h,), c=0., name='conv_relu_1x1_2_b', target='dev0')
        
        #softmax
        self.tparams['conv_logits_W'] = he(shape=[3,two_h,1,1], fan_in=(1*1*two_h), name='conv_logits_W', target='dev0')
        self.tparams['conv_logits_b'] = constant(shape=(3,), c=0., name='conv_logits_b', target='dev0')

    def save_params(self, name):
        #return self.saver.save(self.sess, name)
        self.save(name)

    def load_params(self, path):
        #self.saver.restore(self.sess, path)
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
                #_, l = self.sess.run([self.train_op, self.loss], {self.image: image})
                #self.image.set_value(image)
                #self.gpu_dataset.set_value(image)
                l = self.train_op(image)
                #updates = self.updater(self.params, self.loss)
                train_losses.append(l)
                print 'training...', i, np.mean(train_losses), '\r',

            np.random.shuffle(vidx)

            for j in xrange(0, len(vidx)/5, mbsz):
                image = valid[vidx[i:i+mbsz]]
                l, = self.sess.run([self.loss], {self.image: image})
                validation_losses.append(l)
                print 'validating...', j, np.mean(validation_losses), '\r',

            path = self.save_params("{}/model".format(name))
            self.sample('{}/sample_{}.png'.format(name, e), train[tidx[:16]])
            print "epoch: {}/{}, train loss: {}, validation loss: {}, model saved: {}".format(e,
                    nb_epochs, np.mean(train_losses), np.mean(validation_losses), path)

            del train_losses
            del validation_losses

    def sample(self, name, image, n=4):
        image[:, 16:48, 16:48, :] = 0.
        for i in xrange(16, 48):
            for j in xrange(16, 48):
                #pixel, = self.sess.run([self.preds[:, i, j, :]], {self.image: image})
                pixel = self.pred_fn(image)[:,:,i,j]
                image[:, i, j, :] = pixel.dimshuffle(0,2,3,1)
                print 'sampling...', i, j, '\r',

        canvas = Image.new('RGB', (72*n, 72*n))
        for i in xrange(n):
            for j in xrange(n):
                im = Image.fromarray(np.cast[np.uint8](image[i*n+j, :, :, :] * 255))
                canvas.paste(im, (72*i+4, 72*j+4))
        canvas.save(name)


    def run(self):
        #model = PixelCNN()

        expname = self.experiment_name #sys.argv[1]
        datahome = 'C:\\Users\\Mariane\\Documents\\ift6266\\ift6266-h17-project\\'
        train = np.load(datahome + 'images.train.npz').items()[0][1] / 255.
        valid = np.load(datahome + 'images.valid.npz').items()[0][1] / 255.
        train = train.astype('float32')
        valid = valid.astype('float32')

        #if not os.path.exists(expname):
        #    os.makedirs(expname)
        #else:
        #    ckpt_file = "{}\\saved\\model.pkl".format(expname)
        #    self.load_params(ckpt_file)

        self.train(expname, train, valid)
        
    # -----------------------
    # --- Private Methods ---
    # -----------------------
    def _get_default_hparams(self):
        """ Returns default hyperparameters"""
        return {
            'adam_leanring_rate':   1e-5,
            'init_conv_kernel_size':[7,7],
            'h':                    128,
            'mini_batch_size':      32,
            'nb_epochs':            200,
            'nb_residual_block':    12,
            'nb_relu_units':        32,
            'z_dim':                64
        }

    def _get_random_hparams(self):
        # TODO: Fix random hparams selection
        hparams = {}
        return hparams
