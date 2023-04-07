import copy

import numpy as np
import matplotlib.pyplot as plt
from RPCL4KM import RPCLK
from utils import *
import argparse
import warnings
warnings.filterwarnings('ignore')

def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--k', type=int, default=6)
    parser.add_argument('--seed', default=1141, type=int)
    parser.add_argument('--beta', default=0.3, type=float)
    return parser.parse_args()

if __name__ == '__main__':
    args = get_config()
    np.random.seed(args.seed)
    # generate dataset with 3 clusters
    X = data_gen4RPCLK()
    A = RPCLK(X, args.k, args.beta, args.seed)
    centroids, labels, iters= A.run()

    # Run k-means clustering with RPCL and k=4
    plot4RPCLK(args.k, iters, X, labels, centroids)

