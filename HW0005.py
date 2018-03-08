import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-10., 10., 50)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def deriv_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))


def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
def deriv_tanh(x):
    return 1 - np.square(tanh(x))


def ReLU(x):
    return np.where(x > 0, x, 0)
def deriv_ReLU(x):
    return np.where(x > 0, 1, 0)


def leaky_ReLU(x):
    return np.where(x > 0.01 * x, x, 0.01 * x)
def deriv_leaky_ReLU(x):
    return np.where(x > 0.01 * x, 1, 0.01)


def ELU(x, alpha):
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))
def deriv_ELU(x, alpha):
    return np.where(x > 0, 1, ELU(x, alpha) + alpha)

fig, ax = plt.subplots(1, 2)
ax[0].spines['left'].set_position('center')
ax[0].spines['bottom'].set_position('center')
ax[0].spines['right'].set_position('center')
ax[0].spines['top'].set_position('center')
ax[0].plot(x, sigmoid(x))

ax[1].spines['left'].set_position('center')
ax[1].spines['right'].set_position('center')
ax[1].spines['top'].set_color('none')
ax[1].xaxis.set_ticks_position('bottom')
ax[1].plot(x, deriv_sigmoid(x))
