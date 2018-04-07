import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import sleep


class Clusters:
    def __init__(self, X, k, e, max_it=100):
        self.X = X
        self.k = k
        self.e = e

        self.means = X[np.random.choice(len(X), k)]

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)

        for i in range(max_it):
            self.plot(i)
            self.clusters = self.veronoi(self.means)
            new_means = [self.mean(c) for (m, c) in self.clusters]
            deltas = [self.l2(m, new_m)
                      for m, new_m in zip(self.means, new_means)]
            if np.max(deltas) < self.e:
                print('converged in {} iterations'.format(i))
                break
            self.means = new_means
            sleep(.5)


    @staticmethod
    def l2(a, b):
        return np.sqrt(np.sum(np.power(a - b, 2)))

    def mean(self, cluster):
        cluster = self.X[cluster]
        return np.mean(cluster, axis=0)


    def veronoi(self, means):
        d = np.zeros((len(means), len(self.X)))
        for i, mean in enumerate(means):
            d[i, :] = [self.l2(mean, x) for x in self.X]

        clusters = {m_idx: [] for m_idx in range(len(means))}

        for j, x in enumerate(self.X):
            d_x = d[:, j]
            m_nearest_idx = np.argmin(d_x)
            clusters[m_nearest_idx].append(j)

        return [(means[m_idx], c) for m_idx, c in clusters.items()]

    def plot(self, i):
        self.ax.clear()
        self.ax.set_title('iteration: {}'.format(i))
        ax = self.fig.add_subplot(111)
        veronoi = self.veronoi(self.means)
        for i, (mean, cluster) in enumerate(veronoi):
            c = 'C{}'.format(i % 9)
            cluster = self.X[cluster]
            ax.scatter(cluster[:, 0], cluster[:, 1], color=c, s=1)
            ax.scatter([mean[0]], [mean[1]], color=c, s=100)
        self.fig.show()




if __name__ == '__main__':
    df = pd.read_csv('iris.csv')
    print(df.columns)
    df_length = df[['SepalLengthCm', 'PetalLengthCm']]
    # df_length = df[['SepalLengthCm', 'PetalLengthCm',
    #                 'SepalWidthCm', 'PetalWidthCm']]
    df['Species'].unique()
    clusters = Clusters(df_length.values, k=3, e=.01)
