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
