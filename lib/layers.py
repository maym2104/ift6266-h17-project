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
# Modified from https://github.com/Newmu/dcgan_code/blob/master/lib/ops.py
# MIT License
from math import floor
import theano
import theano.tensor as T
import theano.tensor.signal.pool as T_pool
from .rng import t_rng


# ------------------------
# Activation Layers
# ------------------------
def linear(x):
    return x

def relu(x):
    return (x + abs(x)) / 2.

def clipped_relu(x, max=10.):
    return T.clip((x + abs(x)) / 2., min=0., max=max)

def leaky_relu(x, alpha=0.2):
    return ((1 + alpha) * x + (1 - alpha) * abs(x)) / 2.
lrelu = leaky_relu

def prelu(x, alpha):
    alpha = alpha.dimshuffle('x', 0, 'x', 'x') if x.dim == 4 else alpha
    return leaky_relu(x, alpha)

def sigmoid(x):
    return T.nnet.sigmoid(x)

def fast_sigmoid(x):
    return T.nnet.ultra_fast_sigmoid(x)

def hard_sigmoid(x):
    return T.nnet.hard_sigmoid(x)

def tanh(x):
    return T.tanh(x)

def hard_tanh(x):
    return T.clip(x, min=-1., max=1.)

def softplus(x):
    return T.nnet.softplus(x)

def softmax(x):
    return T.nnet.softmax(x)

def elu(x, alpha=1.):
    return T.nnet.elu(x, alpha)

def maxout(x, nb_pool):
    if x.ndim == 2:
        x = T.max([x[:, n::nb_pool] for n in range(nb_pool)], axis=0)
    elif x.ndim == 4:
        x = T.max([x[:, n::nb_pool, :, :] for n in range(nb_pool)], axis=0)
    else:
        raise NotImplementedError
    return x


# ------------------------
# Cost Layers
# ------------------------
def CategoricalCrossEntropy(y_pred, y_true):
    return T.nnet.categorical_crossentropy(y_pred, y_true).mean()

def BinaryCrossEntropy(y_pred, y_true):
    return T.nnet.binary_crossentropy(y_pred, y_true).mean()

def MeanSquaredError(y_pred, y_true):
    return T.sqr(y_pred - y_true).mean()

def MeanAbsoluteError(y_pred, y_true):
    return T.abs_(y_pred - y_true).mean()

def SquaredHinge(y_pred, y_true):
    return T.sqr(T.maximum(1. - y_true * y_pred, 0.)).mean()

def SquaredError(y_pred, y_true):
    return T.sqr(y_pred - y_true).sum()

def Hinge(y_pred, y_true):
    return T.maximum(1. - y_true * y_pred, 0.).mean()

cce = CCE = CategoricalCrossEntropy
bce = BCE = BinaryCrossEntropy
mse = MSE = MeanSquaredError
mae = MAE = MeanAbsoluteError


# ------------------------
# Regular Layers
# ------------------------
def l2normalize(x, axis=1, e=1e-8, keepdims=True):
    return x/l2norm(x, axis=axis, e=e, keepdims=keepdims)

def l2norm(x, axis=1, e=1e-8, keepdims=True):
    return T.sqrt(T.sum(T.sqr(x), axis=axis, keepdims=keepdims) + e)

def cosine(x, y):
    d = T.dot(x, y.T)
    d /= l2norm(x).dimshuffle(0, 'x')
    d /= l2norm(y).dimshuffle('x', 0)
    return d

def euclidean(x, y, e=1e-8):
    xx = T.sqr(T.sqrt((x*x).sum(axis=1) + e))
    yy = T.sqr(T.sqrt((y*y).sum(axis=1) + e))
    dist = T.dot(x, y.T)            # sqrt(x^2 - 2x*y + y^2)
    dist *= -2
    dist += xx.dimshuffle(0, 'x')
    dist += yy.dimshuffle('x', 0)
    dist = T.sqrt(dist)
    return dist

def concat(tensor_list, axis=0):
    return T.concatenate(tensor_list, axis)

def pool(X, ws, ignore_border=None, stride=None, pad=(0,0), mode='max'):
    """ Generic pooling layer
<<<<<<< HEAD
    X - input (N-D theano tensor of input images) - Input images. Max pooling will be done over the 2 last dimensions.
    ws (tuple of length 2) - Factor by which to downscale (vertical ws, horizontal ws). (2,2) will halve the image in each dimension.
    ignore_border (bool (default None, will print a warning and set to False)) - When True, (5,5) input with ws=(2,2) will generate a (2,2) output. (3,3) otherwise.
    stride (tuple of two ints) - Stride size, which is the number of shifts over rows/cols to get the next pool region. If stride is None, it is considered equal to ws (no overlap on pooling regions).
    pad (tuple of two ints) - (pad_h, pad_w), pad zeros to extend beyond four borders of the images, pad_h is the size of the top and bottom margins, and pad_w is the size of the left and right margins.
    mode ({'max', 'sum', 'average_inc_pad', 'average_exc_pad'}) - Operation executed on each window. max and sum always exclude the padding in the computation. average gives you the choice to include or exclude it.
=======
    X - input (N-D theano tensor of input images) – Input images. Max pooling will be done over the 2 last dimensions.
    ws (tuple of length 2) – Factor by which to downscale (vertical ws, horizontal ws). (2,2) will halve the image in each dimension.
    ignore_border (bool (default None, will print a warning and set to False)) – When True, (5,5) input with ws=(2,2) will generate a (2,2) output. (3,3) otherwise.
    stride (tuple of two ints) – Stride size, which is the number of shifts over rows/cols to get the next pool region. If stride is None, it is considered equal to ws (no overlap on pooling regions).
    pad (tuple of two ints) – (pad_h, pad_w), pad zeros to extend beyond four borders of the images, pad_h is the size of the top and bottom margins, and pad_w is the size of the left and right margins.
    mode ({'max', 'sum', 'average_inc_pad', 'average_exc_pad'}) – Operation executed on each window. max and sum always exclude the padding in the computation. average gives you the choice to include or exclude it.
>>>>>>> 28281d9b5f6c9bf2ddfffd79fbfcfc33c3fb8f41
    """
    if X.ndim >= 2 and len(ws) == 2:
        return T_pool.pool_2d(input=X, ws=ws, ignore_border=ignore_border, stride=stride, pad=pad, mode=mode)
    else:
        raise NotImplementedError

def max_pool_1d(X, ws=2, ignore_border=True, stride=None, pad=0):
    """ Max pooling layer in 1 dimension - see pool() for details """
    input = X.dimshuffle(0, 1, 2, 'x')
    ws = (ws, 1)
    stride = ws if stride is None else (stride, 1)
    pad = (pad, 0)
    pooled = pool(input, ws=ws, ignore_border=ignore_border, stride=stride, pad=pad, mode='max')
    return pooled[:, :, :, 0]

def max_pool(X, ws=(2,2), ignore_border=True, stride=None, pad=(0,0)):
    """ Max pooling layer - see pool() for details """
    return pool(X, ws=ws, ignore_border=ignore_border, stride=stride, pad=pad, mode='max')

def avg_pool(X, ws=(2,2), ignore_border=True, stride=None, pad=(0,0)):
    """ Average pooling layer - see pool() for details """
    return pool(X, ws=ws, ignore_border=ignore_border, stride=stride, pad=pad, mode='average_inc_pad')

def unpool(X, us):
    """ Unpooling layer
    X - input (N-D theano tensor of input images)
    us (tuple of length >= 1) - Factor by which to upscale (vertical ws, horizontal ws). (2,2) will double the image in each dimension. - Factors are applied to last dimensions of X
    """
    x_dims = X.ndim
    output = X
    for i, factor in enumerate(us[::-1]):
        if factor > 1:
            output = T.repeat(output, factor, x_dims - i - 1)
    return output

# Luke Perforated Upsample
# Source: http://www.brml.org/uploads/tx_sibibtex/281.pdf
# Code from: https://gist.github.com/kastnerkyle/f3f67424adda343fef40
def perforated_upsample(X, factor):
    output_shape = [X.shape[1], X.shape[2] * factor, X.shape[3] * factor]
    stride = X.shape[2]
    offset = X.shape[3]
    in_dim = stride * offset
    out_dim = in_dim * factor * factor
    upsamp_matrix = T.zeros((in_dim, out_dim))
    rows = T.arange(in_dim)
    cols = T.cast(rows * factor + (rows / stride * factor * offset), 'int32')
    upsamp_matrix = T.set_subtensor(upsamp_matrix[rows, cols], 1.)
    flat = T.reshape(X, (X.shape[0], output_shape[0], X.shape[2] * X.shape[3]))
    up_flat = T.dot(flat, upsamp_matrix)
    upsamp = T.reshape(up_flat, (X.shape[0], output_shape[0], output_shape[1], output_shape[2]))
    return upsamp

def repeat_upsample(X, factor):
    return unpool(X, us=(factor, factor))

def bilinear_upsample(X, factor):
   return theano.tensor.nnet.abstract_conv.bilinear_upsampling(X, factor, batch_size=X.shape[0], num_input_channels=X.shape[1])

def batchnorm(X, g=None, b=None, u=None, s=None, a=1., e=1e-8):
    """
    batchnorm with support for not using scale and shift parameters
    as well as inference values (u and s) and partial batchnorm (via a)
    will detect and use convolutional or fully connected version
    """
    if X.ndim == 4:
        if u is not None and s is not None:
            _u = u.dimshuffle('x', 0, 'x', 'x')
            _s = s.dimshuffle('x', 0, 'x', 'x')
        else:
            _u = T.mean(X, axis=[0, 2, 3]).dimshuffle('x', 0, 'x', 'x')
            _s = T.mean(T.sqr(X - _u), axis=[0, 2, 3]).dimshuffle('x', 0, 'x', 'x')
        if a != 1:
            _u = (1. - a) * 0. + a * _u
            _s = (1. - a) * 1. + a * _s
        X = (X - _u) / T.sqrt(_s + e)
        if g is not None and b is not None:
            X = X * g.dimshuffle('x', 0, 'x', 'x') + b.dimshuffle('x', 0, 'x', 'x')
    elif X.ndim == 2:
        if u is None and s is None:
            u = T.mean(X, axis=0)
            s = T.mean(T.sqr(X - u), axis=0)
        if a != 1:
            u = (1. - a) * 0. + a * u
            s = (1. - a) * 1. + a * s
        X = (X - u) / T.sqrt(s + e)
        if g is not None and b is not None:
            X = X * g + b
    else:
        raise NotImplementedError
    return X

def dropout(X, p=0.):
    if p > 0:
        retain_prob = 1 - p
        X *= t_rng.binomial(X.shape, p=retain_prob, dtype=theano.config.floatX)
        X /= retain_prob
    return X

def conv_1d(X, w, b=None, border_mode='valid', subsample=(1,), filter_flip=True):
    if isinstance(border_mode, tuple):
        (border_mode,) = border_mode
    if isinstance(border_mode, int):
        border_mode = (border_mode, 0)
    input = X.dimshuffle(0, 1, 2, 'x')
    filter = w.dimshuffle(0, 1, 2, 'x')
    conved = T.nnet.conv2d(input, filter, subsample=(subsample[0], 1), border_mode=border_mode, filter_flip=filter_flip)
    conved = conved[:, :, :, 0]
    if b is not None:
        conved += b.dimshuffle('x', 0, 'x')
    return conved

def conv(X, w, b=None, border_mode='half', subsample=(1, 1), filter_flip=True):
    """ Generic convolution layer
<<<<<<< HEAD
    X - input (symbolic 4D tensor) - Mini-batch of feature map stacks, of shape (batch size, input channels, input rows, input columns). See the optional parameter input_shape.
    w - filters (symbolic 4D tensor) - Set of filters used in CNN layer of shape (output channels, input channels, filter rows, filter columns). See the optional parameter filter_shape.
    border_mode 'valid', 'full', 'half', int, (int1, int2)
    subsample (tuple of len 2) - Factor by which to subsample the output. Also called strides elsewhere.
    filter_flip (bool) - If True, will flip the filter rows and columns before sliding them over the input. This operation is normally referred to as a convolution, and this is the default. If False, the filters are not flipped and the operation is referred to as a cross-correlation
=======
    X - input (symbolic 4D tensor) – Mini-batch of feature map stacks, of shape (batch size, input channels, input rows, input columns). See the optional parameter input_shape.
    w - filters (symbolic 4D tensor) – Set of filters used in CNN layer of shape (output channels, input channels, filter rows, filter columns). See the optional parameter filter_shape.
    border_mode 'valid', 'full', 'half', int, (int1, int2)
    subsample (tuple of len 2) – Factor by which to subsample the output. Also called strides elsewhere.
    filter_flip (bool) – If True, will flip the filter rows and columns before sliding them over the input. This operation is normally referred to as a convolution, and this is the default. If False, the filters are not flipped and the operation is referred to as a cross-correlation
>>>>>>> 28281d9b5f6c9bf2ddfffd79fbfcfc33c3fb8f41
    """
    output =\
        T.nnet.conv2d(
            input=X,
            filters=w,
            border_mode=border_mode,
            subsample=subsample,
            filter_flip=filter_flip)
    if b is not None:
        output += b.dimshuffle('x', 0, 'x', 'x')
    return output

def deconv(X, X_shape, w, b=None, border_mode='half', subsample=(1, 1), filter_flip=True, a=None, target_size=None):
    """ Generic convolution layer
<<<<<<< HEAD
    X - input (symbolic 4D tensor) - This is the input to the transposed convolution
    w - filters (symbolic 4D tensor) - Set of filters used in CNN layer of shape (nb of channels of X, nb of channels of output, filter rows, filter columns). The first 2 parameters are inversed compared to a normal convolution. (Usually (output channels, input channels))
    border_mode 'valid', 'full', 'half', int, (int1, int2)
    subsample (tuple of len 2) - Factor by which to subsample the output. Also called strides elsewhere.
    filter_flip (bool) - If True, will flip the filter rows and columns before sliding them over the input. This operation is normally referred to as a convolution, and this is the default. If False, the filters are not flipped and the operation is referred to as a cross-correlation
=======
    X - input (symbolic 4D tensor) – This is the input to the transposed convolution
    w - filters (symbolic 4D tensor) – Set of filters used in CNN layer of shape (nb of channels of X, nb of channels of output, filter rows, filter columns). The first 2 parameters are inversed compared to a normal convolution. (Usually (output channels, input channels))
    border_mode 'valid', 'full', 'half', int, (int1, int2)
    subsample (tuple of len 2) – Factor by which to subsample the output. Also called strides elsewhere.
    filter_flip (bool) – If True, will flip the filter rows and columns before sliding them over the input. This operation is normally referred to as a convolution, and this is the default. If False, the filters are not flipped and the operation is referred to as a cross-correlation
>>>>>>> 28281d9b5f6c9bf2ddfffd79fbfcfc33c3fb8f41
    a (tuple of len 2) - Additional padding to add to the transposed convolution to get a specific output size
    input_shape (tuple of len 4) - The size of the variable X
    target_size (tuple of len 2 or 4) - indicates the shape you want to get as a result of the transposed convolution (e.g. (64,64) or (128,3,64,64)).
    """
    # Calculating size of o_prime (output after transposed convolution)
    w_shape = w.get_value().shape
    x_chan, i1, i2 = X_shape[1], X_shape[2], X_shape[3]
    c2, c1, k1, k2 = w_shape[0], w_shape[1], w_shape[2], w_shape[3]         # We are going from c2 (nb of channels of X) to c1 (nb of channels of result) ...
    s1, s2 = subsample[0], subsample[1]
    assert c2 == x_chan

    if border_mode == 'half':
        p1, p2 = floor(k1 / 2.), floor(k2 / 2.)
    elif border_mode == 'full':
        p1, p2 = k1 - 1, k2 - 1
    elif border_mode == 'valid':
        p1, p2 = 0, 0
    elif isinstance(border_mode, tuple):
        p1, p2 = border_mode[0], border_mode[1]
    elif isinstance(border_mode, int):
        p1, p2 = border_mode, border_mode
    else:
        raise NotImplementedError

    # 'a' represents additional padding on top and right edge
    # adjust to modify the shape of the result of the transposed convolution
    if a is None:
        if target_size is not None:
            orig_i1, orig_i2 = target_size[-2], target_size[-1]
            a1 = (orig_i1 + 2 * p1 - k1) % s1
            a2 = (orig_i2 + 2 * p2 - k2) % s2
        else:
            a1, a2 = s1 - 1, s2 - 1
    else:
        a1, a2 = a[0], a[1]
    o_prime1 = int(s1 * (i1 - 1) + a1 + k1 - 2 * p1)
    o_prime2 = int(s2 * (i2 - 1) + a2 + k2 - 2 * p2)

    # Transposed Convolution
    output =\
        T.nnet.abstract_conv.conv2d_grad_wrt_inputs(
            output_grad=X,
            filters=w,
            input_shape=(None, c1, o_prime1, o_prime2),
            border_mode=(p1, p2),
            subsample=(s1, s2),
            filter_flip=filter_flip)
    if b is not None:
        output += b.dimshuffle('x', 0, 'x', 'x')
    return output
