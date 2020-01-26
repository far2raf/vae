import functools
import operator


def reduce_shape(shape):
    return functools.reduce(operator.mul, shape)


def learn(model, data, num_of_epoch):
    for index in range(num_of_epoch):
        # MOCK: shuffle data
        for X, y in data:
            model.learn(X)
    return model


