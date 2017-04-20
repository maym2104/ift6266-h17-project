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
