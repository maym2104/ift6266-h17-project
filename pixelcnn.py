import sys, os
import numpy as np
#import tensorflow as tf
#import tensorflow.contrib.slim as slim
from lib.layers import lrelu, relu, conv, conv_1d, sigmoid, deconv, tanh, batchnorm, concat, max_pool, max_pool_1d, avg_pool, dropout, BinaryCrossEntropy, MeanSquaredError, MeanAbsoluteError
from lib.inits import constant, he, normal, orthogonal, uniform
from lib.utils import castX, shared0s, sharedX, floatX
from PIL import Image
import theano
import theano.tensor as T

class nn:

    @staticmethod
    def conv2d(inputs, num_outputs, kernel_size, mask=None, activation_fn=relu, scope=None, input_shape=None):
        ishape = input_shape if input_shape is not None else inputs.shape.eval()
        shape = (kernel_size[0], kernel_size[1], ishape[-1], num_outputs)
        #weights_initializer = tf.contrib.layers.xavier_initializer()

        #with tf.variable_scope(scope):
            #W = tf.get_variable('W', shape, tf.float32, weights_initializer)
        W = he(shape=shape, fan_in=(kernel_size[0]*kernel_size[1]*num_outputs), target='dev0')
            #b = tf.get_variable('b', num_outputs, tf.float32, tf.zeros_initializer())
        b = T.zeros(shape=num_outputs, dtype='float32')

        if mask:
            mid_x = shape[0]/2
            mid_y = shape[1]/2
            mask_filter = np.ones(shape, dtype=np.bool)
            mask_filter[mid_x, mid_y+1:, :, :] = False
            mask_filter[mid_x+1:, :, :, :] = False

            if mask == 'a':
                mask_filter[mid_x, mid_y, :, :] = False

            W  = W * mask_filter

        #h = tf.add(tf.nn.conv2d(inputs, W, strides=[1,1,1,1], padding='SAME'), b)
        h = conv(inputs, W, b, (1,1), (1,1))

        if activation_fn:
            h = activation_fn(h)

        return h

def residual_block(x, name, mask=None, h_size=0):
    nr_filters = h_size if h_size is not 0 else x.shape.eval()[-1]
    h = nn.conv2d(x, nr_filters/2, [1, 1], scope='conv/1x1/{}/1'.format(name))
    h = nn.conv2d(h, nr_filters/2, [3, 3], mask=mask, scope='conv/3x3/{}'.format(name))
    h = nn.conv2d(h, nr_filters, [1, 1], scope='conv/1x1/{}/2'.format(name))
    return h + x


class PixelCNN:
    mask = np.ones((64, 64, 1), dtype=np.bool)
    mask[16:48, 16:48, :] = False

    def __init__(self):
        self.image = uniform((32,28,28,1), scale=0.05, name='image', shared=True, target=None)
        #self.image = tf.placeholder(tf.float32, [None, 28, 28, 1])
        #self.image = T.tensor4('src_images', dtype='float32')
        #def initNet(input):
        h = nn.conv2d(self.image, 256, [7, 7], mask='a', scope='conv/7x7')
        #conv7x7 = theano.function([self.image], nn.conv2d(self.image, 256, [7, 7], mask='a', scope='conv/7x7', input_shape=(None, 28, 28, 1)), name='conv/7x7')
        for i in xrange(12):
            h = residual_block(h, str(i), mask='b')
        h = nn.conv2d(h, 32, [1, 1], scope='conv/relu/1x1/1')
        h = nn.conv2d(h, 32, [1, 1], scope='conv/relu/1x1/2')
        self.logits = nn.conv2d(h, 3, [1, 1], activation_fn=None, scope='conv/logits')

        self.preds = sigmoid(self.logits)
        self.losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=self.image)
        self.losses = BinaryCrossEntropy(self.logits, self.image)
        self.losses = self.losses[:, 16:48, 16:48, :]
        self.loss = tf.reduce_mean(tf.reduce_sum(self.losses, axis=[1, 2, 3]))
        self.train_op = tf.train.AdamOptimizer(1e-5).minimize(self.loss)
        #self.sess = tf.Session()
        #self.sess.run(tf.global_variables_initializer())
        #f = theano.function([self.image], initNet(self.image))
        
        #self.saver = tf.train.Saver()

    def save_params(self, name):
        return self.saver.save(self.sess, name)

    def load_params(self, path):
        self.saver.restore(self.sess, path)
        print "params restored from {}".format(path)

    def train(self, name, train, valid, mbsz=32, nb_epochs=200):
        tidx = np.arange(len(train))
        vidx = np.arange(len(valid))
		
        #self.image = sharedX(train, dtype='float32', name=None, target='dev0')
        #def initNet(input):
            #h = nn.conv2d(input, 256, [7, 7], mask='a', scope='conv/7x7',input_shape=(None, 28, 28, 1))
            #conv7x7 = theano.function([self.image], nn.conv2d(self.image, 256, [7, 7], mask='a', scope='conv/7x7', input_shape=(None, 28, 28, 1)), name='conv/7x7')
            #for i in xrange(12):
                 #h = residual_block(h, str(i), mask='b', h_size=256)
            #h = nn.conv2d(h, 32, [1, 1], scope='conv/relu/1x1/1', input_shape=[256])
            #h = nn.conv2d(h, 32, [1, 1], scope='conv/relu/1x1/2', input_shape=[32])
            #self.logits = nn.conv2d(h, 3, [1, 1], activation_fn=None, scope='conv/logits', input_shape=[32])

            #self.preds = sigmoid(self.logits)
            #self.losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=self.image)
            #self.losses = BinaryCrossEntropy(self.logits, self.image)
            #self.losses = self.losses[:, 16:48, 16:48, :]
            #self.loss = tf.reduce_mean(tf.reduce_sum(self.losses, axis=[1, 2, 3]))
            #self.train_op = tf.train.AdamOptimizer(1e-5).minimize(self.loss)

        for e in xrange(nb_epochs):
            train_losses = []
            validation_losses = []

            np.random.shuffle(tidx)

            for i in xrange(0, len(tidx)/5, mbsz):
                image = train[tidx[i:i+mbsz]]
                _, l = self.sess.run([self.train_op, self.loss], {self.image: image})
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
                pixel, = self.sess.run([self.preds[:, i, j, :]], {self.image: image})
                image[:, i, j, :] = pixel
                print 'sampling...', i, j, '\r',

        canvas = Image.new('RGB', (72*n, 72*n))
        for i in xrange(n):
            for j in xrange(n):
                im = Image.fromarray(np.cast[np.uint8](image[i*n+j, :, :, :] * 255))
                canvas.paste(im, (72*i+4, 72*j+4))
        canvas.save(name)


if __name__ == '__main__':
    model = PixelCNN()

    expname = sys.argv[1]
    datahome = 'C:\\Users\\Mariane\\Documents\\ift6266\\ift6266-h17-project\\'
    train = np.load(datahome + 'images.train.npz').items()[0][1] / 255.
    valid = np.load(datahome + 'images.valid.npz').items()[0][1] / 255.

    if not os.path.exists(expname):
        os.makedirs(expname)
    else:
        ckpt_file = "{}/model".format(expname)
        model.load_params(ckpt_file)

    model.train(expname, train, valid)
