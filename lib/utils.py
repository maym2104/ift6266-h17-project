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

# Modified from https://raw.githubusercontent.com/Newmu/dcgan_code/master/lib/theano_utils.py
# MIT License
import numpy as np
import os
import theano
import errno

def intX(X):
    return np.asarray(X, dtype=np.int32)

def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)

def castX(X):
    return theano.tensor.cast(X, dtype=theano.config.floatX)

def transfer(X, device):
    if theano.config.device == 'cpu' or theano.config.contexts == '':
        return None
    else:
        return X.transfer(device)

def sharedX(X, dtype=theano.config.floatX, name=None, target=None):
    if theano.config.device == 'cpu' or theano.config.contexts == '':
        return theano.shared(np.asarray(X, dtype=dtype), name=name, borrow=True)
    else:
        return theano.shared(np.asarray(X, dtype=dtype), name=name, borrow=True, target=target)

def shared0s(shape, dtype=theano.config.floatX, name=None, target=None):
    return sharedX(np.zeros(shape), dtype=dtype, name=name, target=target)

def sharedNs(shape, n, dtype=theano.config.floatX, name=None, target=None):
    return sharedX(np.ones(shape)*n, dtype=dtype, name=name, target=target)

# Source: http://stackoverflow.com/questions/600268/mkdir-p-functionality-in-python
def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise
