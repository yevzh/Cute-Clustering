import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.mixture import BayesianGaussianMixture
import numpy as np
import warnings
import argparse
from utils import *

class GMM:
    def __init__(self, data, n_components, method, n_clusters, n_samples):
        self.model = GaussianMixture(n_components=n_components)
        self.n_components = n_components
        self.n_samples = n_samples
        self.n_clusters = n_clusters
        self.data = data
        self.model.fit(self.data)
        self.method = method
        self.aic = []
        self.bic = []
        self.vbem_model = BayesianGaussianMixture(n_components=n_components)
        self.vbem_model.fit(self.data)
        # self.vbem = []

    def aic_evaluate(self):
        for n in range(1, self.n_components+1):
            gmm = GaussianMixture(n_components=n)
            gmm.fit(self.data)
            self.aic.append(gmm.aic(self.data))

    def bic_evaluate(self):
        for n in range(1, self.n_components+1):
            gmm = GaussianMixture(n_components=n)
            gmm.fit(self.data)
            self.bic.append(gmm.bic(self.data))


    def show(self):
        if self.method == 'None':
            plt.figure()
            labels = self.model.predict(self.data)
            plt.scatter(self.data[:, 0], self.data[:, 1], c=labels, s=15)
            plt.show()

        if self.method == 'aic':
            self.aic_evaluate()
            best_n_components = np.argmin(self.aic)+1
            gmm = GaussianMixture(n_components=best_n_components)
            gmm.fit(self.data)
            plt.figure()
            labels = gmm.predict(self.data)
            n_labels = (len(set(labels)))
            method = self.method
            n_samples = self.n_samples
            n_clusters = self.n_clusters
            plt.title(f'{method} sample{n_samples} with {n_clusters}centroids, to {n_labels} clusters'.format(method=method,n_samples=n_samples,n_clusters=n_clusters,n_labels=n_labels ))
            plt.scatter(self.data[:, 0], self.data[:, 1], c=labels, s=15)
            plt.savefig(f'{method} sample{n_samples} with {n_clusters}centroids, to {n_labels} clusters.png'.format(method=method,n_samples=n_samples,n_clusters=n_clusters,n_labels=n_labels ))
            plt.show()
            plt.close()
            # print(self.aic)

        if self.method == 'bic':
            self.bic_evaluate()
            best_n_components = np.argmin(self.bic)+1
            gmm = GaussianMixture(n_components=best_n_components)
            gmm.fit(self.data)
            plt.figure()
            labels = gmm.predict(self.data)
            n_labels = (len(set(labels)))
            method = self.method
            n_samples = self.n_samples
            n_clusters = self.n_clusters
            plt.title(f'{method} sample{n_samples} with {n_clusters}centroids, to {n_labels} clusters'.format(method=method,n_samples=n_samples,n_clusters=n_clusters,n_labels=n_labels ))
            plt.scatter(self.data[:, 0], self.data[:, 1], c=labels, s=15)
            plt.savefig(f'{method} sample{n_samples} with {n_clusters}centroids, to {n_labels} clusters.png'.format(method=method,n_samples=n_samples,n_clusters=n_clusters,n_labels=n_labels ))
            plt.show()
            plt.close()
            # print(self.bic)

        if self.method == 'vbem':
            plt.figure()
            labels = self.vbem_model.predict(self.data)
            n_labels = (len(set(labels)))
            method = self.method
            n_samples = self.n_samples
            n_clusters = self.n_clusters
            plt.title(f'{method} sample{n_samples} with {n_clusters}centroids, to {n_labels} clusters'.format(method=method,n_samples=n_samples,n_clusters=n_clusters,n_labels=n_labels ))
            plt.scatter(self.data[:, 0], self.data[:, 1], c=labels, s=15)
            plt.savefig(f'{method} sample{n_samples} with {n_clusters}centroids, to {n_labels} clusters.png'.format(method=method,n_samples=n_samples,n_clusters=n_clusters,n_labels=n_labels ))
            plt.show()
            plt.close()

