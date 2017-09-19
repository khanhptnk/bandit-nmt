import numpy as np
import random

def _adver(rs, _not_use):
    return [1 - r for r in rs]

def _random(rs, _not_use):
    return [random.random() for i in xrange(len(rs))]

def _bin(rs, b):
    return [round(r * b) / b for r in rs]

def _variance(rs, scale):
    res = []
    for r in rs:
        # Use 0.67 instead of 67 because scores are in [0,1] instead of [0,100] as in human eval data.
        std = min(r * 0.64, -0.67 * r + 0.67) * scale
        r_new = np.random.normal(r, std)
        r_new = max(0., min(r_new, 1.))
        res.append(r_new)
    return res

#def _noise(rs, std):
#    noises = np.random.normal(0, std, size=len(rs)).tolist()
#    return [r + noise for r, noise in zip(rs, noises)]

def _curve(rs, p):
    return [r**p for r in rs]

class PertFunction(object):
    def __init__(self, func_name, param):
        self.param = param
        if func_name == "bin":
            self.func = _bin
        elif func_name == "skew":
            self.func = _skew
        elif func_name == "variance":
            self.func = _variance
        elif func_name == "random":
            self.func = _random
        elif func_name == "adver":
            self.func = _adver

    def __call__(self, r):
        return self.func(r, self.param)
