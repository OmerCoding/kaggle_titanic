import numpy as np
import numpy.random as r
from sklearn.preprocessing import StandardScaler
import matplotlib.pylab as plt

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
    w = {}
    b = {}
    for l in range(1, len(nn_structure)):
        w[l] = r.random_sample((nn_structure[l], nn_structure[l-1]))
        b[l] = r.random_sample((nn_structure[l],))
    return w, b


def init_tri_values(nn_structure):
    tri_w = {}
    tri_b = {}
    for l in range(1, len(nn_structure)):
        tri_w[l] = np.zeros((nn_structure[l], nn_structure[l-1]))
        tri_b[l] = np.zeros((nn_structure[l],))
    return tri_w, tri_b


def feed_forward(x, w, b):
    h = {1: x}
    z = {}
    for l in range(1, len(w) + 1):
        if l == 1:
            node_in = x
        else:
            node_in = h[l]
        z[l+1] = w[l].dot(node_in) + b[l]
        h[l+1] = sigma(z[l+1])
    return h, z


def calculate_out_layer_delta(y, h_out, z_out):
    return -(y-h_out) * sigma_deriv(z_out)


def calculate_hidden_delta(delta_plus_1, w_l, z_l):
    return np.dot(np.transpose(w_l), delta_plus_1) * sigma_deriv(z_l)


def train_nn(nn_structure, X, y, iter_num=3000, alpha=0.25):
    w, b = setup_and_init_weights(nn_structure)
    cnt = 0
    m = len(y)
    avg_cost_func = []
    print('Starting gradient descent for {} iterations'.format(iter_num))
    while cnt < iter_num:
        if cnt%1000 == 0:
            print('Iteration {} of {}'.format(cnt, iter_num))
        tri_w, tri_b = init_tri_values(nn_structure)
        avg_cost = 0
        for i in range(len(y)):
            delta = {}
            h, z = feed_forward(X[i, :], w, b)
            for l in range(len(nn_structure), 0, -1):
                if l == len(nn_structure):
                    delta[l] = calculate_out_layer_delta(y[i], h[l], z[l])
                    avg_cost += np.linalg.norm((y[i] - h[l]))
                else:
                    if l > 1:
                        delta[l] = calculate_hidden_delta(delta[l+1], w[l], z[l])
                    tri_w[l] += np.dot(delta[l+1][:, np.newaxis], np.transpose(h[l][:, np.newaxis]))
                    tri_b[l] += delta[l+1]
        for l in range(len(nn_structure) - 1, 0, -1):
            w[l] += -alpha * (1.0/m * tri_w[l])
            b[l] += -alpha * (1.0/m * tri_b[l])
        avg_cost = 1.0/m * avg_cost
        avg_cost_func.append(avg_cost)
        cnt += 1
    return w, b, avg_cost_func


w, b, avg_cost_funct = train_nn(nn_structure, X, y)


plt.plot(avg_cost_funct)
plt.ylabel('Average J')
plt.xlabel('Iteration number')
plt.show()


print(X.size)