# Modified from https://raw.githubusercontent.com/Newmu/dcgan_code/master/lib/rng.py
# MIT License
import numpy as np
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import theano.tensor.shared_randomstreams
from random import Random

seed = 42

py_rng = Random(seed)
np_rng = np.random.RandomState(seed)
t_rng = RandomStreams(seed)
t_rng_2 = theano.tensor.shared_randomstreams.RandomStreams(seed)

def set_seed(n):
    global seed, py_rng, np_rng, t_rng

    seed = n
    py_rng = Random(seed)
    np_rng = np.random.RandomState(seed)

t_rng = RandomStreams(seed)
