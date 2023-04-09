import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets._samples_generator import make_blobs
def data_gen4RPCLK():
    X = np.concatenate([
        np.random.multivariate_normal([2, 2], [[0.2, 0], [0, 0.5]], size=100),
        np.random.multivariate_normal([-2, 2], [[0.2, 0], [0, 0.5]], size=100),
        np.random.multivariate_normal([0, -2], [[0.2, 0], [0, 0.5]], size=100)
    ])
    return X

def plot4RPCLK(num_k, num_iters,X, labels, centroids, beta):
    plt.title(f'RPCL k={num_k} after {num_iters} iterations, gamma={beta}'.format(num_k=num_k, num_iters=num_iters,beta=beta) )
    plt.scatter(X[:, 0], X[:, 1], c=labels)
    plt.scatter(centroids[:, 0], centroids[:, 1], s=100, marker='x', c='r')

    plt.savefig(fname=f'RPCL k={num_k} after {num_iters} iterations, gamma={beta}.png'.format(num_k=num_k, num_iters=num_iters,beta=beta))
    plt.show()
def data_gen4GMM(num_clusters=3, num_samples=1000, cluster_std= 0.5):
    X, y_true = make_blobs(n_samples=num_samples, centers=num_clusters, cluster_std=cluster_std, random_state=0)
    return X