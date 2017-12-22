import numpy as np
import numpy.random as r
from sklearn.preprocessing import StandardScaler

data = np.loadtxt(open("train.csv", "rb"), delimiter=",", skiprows=1)
y = np.loadtxt(open("results.csv", "rb"), delimiter=",", skiprows=1)

X_scale = StandardScaler()
X = X_scale.fit_transform(data)

nn_structure = [10,10,1]


def sigma(x):
    return 1 / (1 + np.exp(-x))


def sigma_deriv(x):
    return sigma(x) * (1 - sigma(x))


def setup_and_init_weights(nn_structure):
    W = {}
    b = {}
    for l in range(1, len(nn_structure)):
        W[l] = r.random_sample((nn_structure[l], nn_structure[l-1]))
        b[l] = r.random_sample((nn_structure[l],))
    return W, b


def init_tri_values(nn_structure):
    tri_w = {}
    tri_b = {}
    for l in range(1, len(nn_structure)):
        tri_w[l] = np.zeros((nn_structure[l], nn_structure[l-1]))
        tri_b[l] = np.zeros((nn_structure[l],))
    return tri_w, tri_b



print(X.size)