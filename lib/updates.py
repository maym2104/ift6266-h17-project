# Modified from https://raw.githubusercontent.com/Newmu/dcgan_code/master/lib/updates.py
# MIT License
import theano
import theano.tensor as T
from .utils import floatX
from .layers import l2norm

# ------------------------
# Regularization
# ------------------------
def clip_norm(grad, clip, norm):
    if clip > 0:
        grad = T.switch(T.ge(norm, clip), grad * clip / norm, grad)
    return grad

def clip_norms(grads, clip):
    norm = T.sqrt(sum([T.sum(grad ** 2) for grad in grads]))
    return [clip_norm(grad, clip, norm) for grad in grads]

# Base regularizer
class Regularizer(object):
    def __init__(self, l1=0., l2=0., maxnorm=0., l2norm=False, frobnorm=False):
        self.__dict__.update(locals())

    def max_norm(self, param, maxnorm):
        if maxnorm > 0:
            norms = T.sqrt(T.sum(T.sqr(param), axis=0))
            desired = T.clip(norms, 0, maxnorm)
            param = param * (desired / (1e-7 + norms))
        return param

    def l2_norm(self, param):
        return param / l2norm(param, axis=0)

    def frob_norm(self, param, nrows):
        return (param / T.sqrt(T.sum(T.sqr(param)))) * T.sqrt(nrows)

    def gradient_regularize(self, param, grad):
        grad += param * self.l2
        grad += T.sgn(param) * self.l1
        return grad

    def weight_regularize(self, param):
        param = self.max_norm(param, self.maxnorm)
        if self.l2norm:
            param = self.l2_norm(param)
        if self.frobnorm > 0:
            param = self.frob_norm(param, self.frobnorm)
        return param

# ------------------------
# Updates
# ------------------------
class Update(object):
    def __init__(self, regularizer=Regularizer(), clipnorm=0.):
        self.__dict__.update(locals())

    def __call__(self, params, grads):
        raise NotImplementedError

# Stochastic Gradient Descent
class SGD(Update):
    def __init__(self, lr=0.01, *args, **kwargs):
        Update.__init__(self, *args, **kwargs)
        self.__dict__.update(locals())

    def __call__(self, params, cost):
        updates = []
        grads = T.grad(cost, params)
        grads = clip_norms(grads, self.clipnorm)
        for param, grad in zip(params, grads):
            grad = self.regularizer.gradient_regularize(param, grad)
            updated_param = param - self.lr * grad
            updated_param = self.regularizer.weight_regularize(updated_param)
            updates.append((param, updated_param))
        return updates

# SGD with momentum
class Momentum(Update):
    def __init__(self, lr=0.01, momentum=0.9, *args, **kwargs):
        Update.__init__(self, *args, **kwargs)
        self.__dict__.update(locals())

    def __call__(self, params, cost):
        updates = []
        grads = T.grad(cost, params)
        grads = clip_norms(grads, self.clipnorm)
        for param, grad in zip(params, grads):
            grad = self.regularizer.gradient_regularize(param, grad)
            m = theano.shared(param.get_value() * 0.)
            v = (self.momentum * m) - (self.lr * grad)
            updates.append((m, v))

            updated_param = param + v
            updated_param = self.regularizer.weight_regularize(updated_param)
            updates.append((param, updated_param))
        return updates

# SGD with Nesterov Accelerated Gradient
class Nesterov(Update):
    def __init__(self, lr=0.01, momentum=0.9, *args, **kwargs):
        Update.__init__(self, *args, **kwargs)
        self.__dict__.update(locals())

    def __call__(self, params, cost):
        updates = []
        grads = T.grad(cost, params)
        grads = clip_norms(grads, self.clipnorm)
        for param, grad in zip(params, grads):
            grad = self.regularizer.gradient_regularize(param, grad)
            m = theano.shared(param.get_value() * 0.)
            v = (self.momentum * m) - (self.lr * grad)

            updated_param = param + self.momentum * v - self.lr * grad
            updated_param = self.regularizer.weight_regularize(updated_param)
            updates.append((m, v))
            updates.append((param, updated_param))
        return updates

# RMS Prop
class RMSprop(Update):
    def __init__(self, lr=0.001, rho=0.9, epsilon=1e-6, *args, **kwargs):
        Update.__init__(self, *args, **kwargs)
        self.__dict__.update(locals())

    def __call__(self, params, cost):
        updates = []
        grads = T.grad(cost, params)
        grads = clip_norms(grads, self.clipnorm)
        for param, grad in zip(params, grads):
            grad = self.regularizer.gradient_regularize(param, grad)
            acc = theano.shared(param.get_value() * 0.)
            acc_new = self.rho * acc + (1 - self.rho) * grad ** 2
            updates.append((acc, acc_new))

            updated_param = param - self.lr * (grad / T.sqrt(acc_new + self.epsilon))
            updated_param = self.regularizer.weight_regularize(updated_param)
            updates.append((param, updated_param))
        return updates

# Adam
class Adam(Update):
    def __init__(self, lr=0.001, b1=0.9, b2=0.999, e=1e-8, l=1 - 1e-8, *args, **kwargs):
        Update.__init__(self, *args, **kwargs)
        self.__dict__.update(locals())

    def __call__(self, params, cost):
        updates = []
        grads = T.grad(cost, params)
        grads = clip_norms(grads, self.clipnorm)
        t = theano.shared(floatX(1.))
        b1_t = self.b1 * self.l ** (t - 1)

        for param, grad in zip(params, grads):
            grad = self.regularizer.gradient_regularize(param, grad)
            m = theano.shared(param.get_value() * 0.)
            v = theano.shared(param.get_value() * 0.)

            m_t = b1_t * m + (1 - b1_t) * grad
            v_t = self.b2 * v + (1 - self.b2) * grad ** 2
            m_c = m_t / (1 - self.b1 ** t)
            v_c = v_t / (1 - self.b2 ** t)
            p_t = param - (self.lr * m_c) / (T.sqrt(v_c) + self.e)
            p_t = self.regularizer.weight_regularize(p_t)
            updates.append((m, m_t))
            updates.append((v, v_t))
            updates.append((param, p_t))
        updates.append((t, t + 1.))
        return updates

# AdaGrad
class Adagrad(Update):
    def __init__(self, lr=0.01, epsilon=1e-6, *args, **kwargs):
        Update.__init__(self, *args, **kwargs)
        self.__dict__.update(locals())

    def __call__(self, params, cost):
        updates = []
        grads = T.grad(cost, params)
        grads = clip_norms(grads, self.clipnorm)
        for param, grad in zip(params, grads):
            grad = self.regularizer.gradient_regularize(param, grad)
            acc = theano.shared(param.get_value() * 0.)
            acc_t = acc + grad ** 2
            updates.append((acc, acc_t))

            p_t = param - (self.lr / T.sqrt(acc_t + self.epsilon)) * grad
            p_t = self.regularizer.weight_regularize(p_t)
            updates.append((param, p_t))
        return updates

# AdeDelta
class Adadelta(Update):
    def __init__(self, lr=0.5, rho=0.95, epsilon=1e-6, *args, **kwargs):
        Update.__init__(self, *args, **kwargs)
        self.__dict__.update(locals())

    def __call__(self, params, cost):
        updates = []
        grads = T.grad(cost, params)
        grads = clip_norms(grads, self.clipnorm)
        for param, grad in zip(params, grads):
            grad = self.regularizer.gradient_regularize(param, grad)

            acc = theano.shared(param.get_value() * 0.)
            acc_delta = theano.shared(param.get_value() * 0.)
            acc_new = self.rho * acc + (1 - self.rho) * grad ** 2
            updates.append((acc, acc_new))

            update = grad * T.sqrt(acc_delta + self.epsilon) / T.sqrt(acc_new + self.epsilon)
            updated_param = param - self.lr * update
            updated_param = self.regularizer.weight_regularize(updated_param)
            updates.append((param, updated_param))

            acc_delta_new = self.rho * acc_delta + (1 - self.rho) * update ** 2
            updates.append((acc_delta, acc_delta_new))
        return updates

# No updates
class NoUpdate(Update):
    def __init__(self, lr=0.01, momentum=0.9, *args, **kwargs):
        Update.__init__(self, *args, **kwargs)
        self.__dict__.update(locals())

    def __call__(self, params, cost):
        updates = []
        for param in params:
            updates.append((param, param))
        return updates
