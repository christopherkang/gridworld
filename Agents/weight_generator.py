import numpy as np


def random_uniform(shape, low, high):
    assert high - low > 0
    spread = high - low
    output = np.random.rand(shape) * spread + low
    return output


def constant(shape, value):
    output = np.zeros(shape) + value
    return output


_GENERATION_TYPES = {
    "RAND": (random_uniform, "UNIFORM DISTRIBUTION, LOW-HIGH"),
    "CONST": (constant, "CONSTANT DISTRIBUTION, VALUE")
}
