import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import sleep


class Clusters:
    def __init__(self, X, k, e, max_it=100):
        self.X = X
        self.k = k
        self.e = e
        self.its = 0

        self.means = X[np.random.choice(len(X), k)]

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(121)

        while self.its <= max_it:
            # self.plot(i)
            self.clusters = self.veronoi(self.means)
            new_means = [self.mean(c) for m, c in self.clusters]
            deltas = [self.l2(m, new_m)
                      for m, new_m in zip(self.means, new_means)]
            if np.max(deltas) < self.e:
                print('converged in {} iterations'.format(self.its))
                break
            self.its += 1
            self.means = new_means
            sleep(.5)

        self.Y = np.zeros(len(X), dtype=int)
        for i, (mean, cluster) in enumerate(self.clusters):
            for x in cluster:
                self.Y[x] = i



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


def plot(ax, X, clusters):
    clusters = sorted(clusters, key=lambda x: x[0][0])
    for i, (mean, cluster) in enumerate(clusters):
        color = 'C{}'.format(i % 9)
        cluster = X[cluster]
        ax.scatter(cluster[:, 0], cluster[:, 1], c=color, s=1)
        ax.scatter([mean[0]], [mean[1]], c=color, s=100)


def plot_actual(ax, df, features):
    clusters = []
    for i, s in enumerate(df.Species.unique()):
        cluster = df[df.Species == s][features].values
        mean = np.mean(cluster, axis=0)
        clusters.append((mean, cluster))

    clusters = sorted(clusters, key=lambda x: x[0][0])
    for i, (mean, cluster) in enumerate(clusters):
        color = 'C{}'.format(i % 9)
        ax.scatter(cluster[:, 0], cluster[:, 1], c=color, s=1)
        ax.scatter([mean[0]], [mean[1]], c=color, s=100)





if __name__ == '__main__':
    df = pd.read_csv('iris.csv')

    features = [
        # 'SepalLengthCm',
        # 'PetalLengthCm',
        'SepalWidthCm',
        'PetalWidthCm'
    ]

    df_length = df[features]

    k = len(df.Species.unique())
    clu = Clusters(df_length.values, k=k, e=.01)


    df['Cluster'] = clu.Y

    df.to_csv('out.csv')

    for s in df.Species.unique():
        for c in range(k):
            count = len(df[(df.Cluster == c) & (df.Species == s)])
            print('species={:20s} cluster={}: {:5d}'.format(s, c, count))


    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    plot(ax1, clu.X, clu.clusters)
    plot_actual(ax2, df, features)
    ax1.set_title('kmeans')
    ax2.set_title('actual')
    fig.show()
