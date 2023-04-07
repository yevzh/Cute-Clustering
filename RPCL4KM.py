import copy
import numpy as np

class RPCLK:
    def __init__(self, X, cluster_num, gamma, random_seed):
        self.data = X
        self.cluster_num = cluster_num
        self.gamma = gamma
        self.random_seed = random_seed
        self.dim = np.shape(self.data)[1]
        self.deltas = []
        self.stale_step = 0
        self.means = np.random.rand(self.cluster_num, self.dim)

    def pkn(self):
        p = np.zeros((len(self.means), self.data.shape[0]))
        for i, point in enumerate(self.data):
            sqrdist = np.sum(np.square(np.array([point]).repeat(len(self.means),axis=0)-self.means), axis=1)
            c = np.argmin(sqrdist)
            p[c][i] = 1
            # find the second best
            # 你不要过来啊
            sqrdist[c] = np.inf
            r = np.argmin(sqrdist)
            p[r][i] = -self.gamma
        return p
    def converge(self, delta):
        self.deltas.append(delta)
        if len(self.deltas) > 1:
            if np.abs(self.deltas[len(self.deltas)-2]-delta)<=0.01:
                self.stale_step += 1
                if self.stale_step > 5:
                    return True
            else:
                self.stale_step = 0
        return False

    def run(self):
        lr = 0.01 * self.cluster_num
        num_iter = 0
        while True:
            num_iter += 1
            prev_cluster_num = len(self.means)
            self.p = self.pkn()
            # EM更新
            best = np.clip(self.p, 0, 1)
            repel = np.clip(self.p, -self.gamma, 0)
            remove_means = []
            old_means = copy.deepcopy(self.means)
            for k in range(len(self.means)):
                if np.sum(best[k])!=0:
                    self.means[k] = np.average(self.data, weights=best[k], axis=0)
                    self.means[k] += (np.sum((self.data - np.array([self.means[k]]).repeat(len(self.data), axis=0)) * np.array([repel[k]]).repeat(self.dim, axis=0).T, axis=0) * lr)
                else:
                    remove_means.append(k)

            delta = np.sum(np.sum(np.square(self.means-old_means),axis=1))
            self.means = np.delete(self.means, remove_means, 0)
            self.p = np.delete(self.p, remove_means, 0)

            if self.converge(delta):
                break
            if len(self.means) != prev_cluster_num:
                lr *= len(self.means) / self.cluster_num

            if len(self.deltas) % len(self.means) == 0:
                lr /= 2
        labels = np.argmin(np.sum((self.data[:, None, :] - self.means[None, :, :]) ** 2, axis=2), axis=1)
        # print(labels)
        return self.means, labels, num_iter




