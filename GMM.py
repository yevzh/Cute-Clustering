
from model_selection import *
import warnings
import argparse
from utils import *


warnings.filterwarnings('ignore')

def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_components', type=int, default=10)
    parser.add_argument('--n_clusters', default=6, type=int)
    parser.add_argument('--n_samples', default=1000, type=int)
    parser.add_argument('--cluster_std', default=0.6, type=float)
    parser.add_argument('--method', default='vbem', type=str)
    return parser.parse_args()


if __name__ == '__main__':
    args = get_config()
    data = data_gen4GMM(args.n_clusters,args.n_samples, args.cluster_std)
    gmm = GMM(data,args.n_components, method=args.method)
    gmm.show()

