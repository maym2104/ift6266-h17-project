# -*- coding: utf-8 -*-
# Modified from https://raw.githubusercontent.com/Newmu/dcgan_code/master/lib/inits.py
# MIT License
from math import sqrt
import numpy as np
from .utils import sharedX
from .rng import np_rng
import theano

def not_shared(X, dtype=theano.config.floatX, name=None, target=None):
    return X

# Initializations
def uniform(shape, scale=0.05, name=None, shared=True, target=None):
    wrapper = sharedX if shared else not_shared
    return wrapper(np_rng.uniform(low=-scale, high=scale, size=shape), name=name, target=target)

def normal(shape, mean=0., std_dev=1., name=None, shared=True, target=None):
    wrapper = sharedX if shared else not_shared
    return wrapper(np_rng.normal(loc=mean, scale=std_dev, size=shape), name=name, target=target)

def orthogonal(shape, scale=1.1, name=None, shared=True, target=None):
    wrapper = sharedX if shared else not_shared
    flat_shape = (shape[0], np.prod(shape[1:]))
    a = np_rng.normal(0.0, 1.0, flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    q = u if u.shape == flat_shape else v  # pick the one with the correct shape
    q = q.reshape(shape)
    return wrapper(scale * q[:shape[0], :shape[1]], name=name, target=target)

def constant(shape, c=0., name=None, shared=True, target=None):
    wrapper = sharedX if shared else not_shared
    return wrapper(np.ones(shape) * c, name=name, target=target)

def glorot(shape, fan_in, fan_out, name=None, shared=True, target=None):
    return uniform(shape, scale=sqrt(12. / (fan_in + fan_out)), name=name, shared=shared, target=target)

def he(shape, fan_in, factor=sqrt(2.), name=None, shared=True, target=None):
    return normal(shape, mean=0., std_dev=(factor * sqrt(1. / fan_in)), name=name, shared=shared, target=target)
