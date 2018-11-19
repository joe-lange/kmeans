import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#were running kmeans!
class KMeans:
    def __init__(self, X, k, e, max_its=50):
        """
        :param X: input data
        :param k: number of clusters
        :param e: convergence threshold
        :param max_its: maximum iterations

        self.clusters: computed clusters
        self.Y: array of labels matching X
        """
        self.X = X
        self.its = 0

        self.clusters = self.compute(k, e, max_its)
        self.Y = self.label(self.clusters)


    def compute(self, k, e, max_its):
        means = self.X[np.random.choice(len(self.X), k)]
        while True:
            clusters = self.veronoi(means)
            new_means = [
                self.mean(member_ids)
                    for mean, member_ids in clusters
            ]
            deltas = [
                self.l2(m, new_m)
                    for m, new_m in zip(means, new_means)
            ]
            means = new_means
            self.its += 1
            if np.max(deltas) < e or self.its == max_its:
                return clusters

    def label(self, clusters):
        Y = np.zeros(len(self.X), dtype=int)
        for cluster_id, (mean, member_ids) in enumerate(clusters):
            for member_id in member_ids:
                Y[member_id] = cluster_id
        return Y


    @staticmethod
    def l2(a, b):
        return np.sqrt(np.sum(np.power(a - b, 2)))

    def mean(self, member_ids):
        cluster = self.X[member_ids]
        return np.mean(cluster, axis=0)


    def veronoi(self, means):
        d = np.zeros((len(means), len(self.X)))
        for i, mean in enumerate(means):
            d[i, :] = [self.l2(mean, x) for x in self.X]

        clusters = {mean_id: [] for mean_id in range(len(means))}

        for j, x in enumerate(self.X):
            d_x = d[:, j]
            nearest_mean_id = np.argmin(d_x)
            clusters[nearest_mean_id].append(j)

        return [
            (means[mean_id], member_ids)
                for mean_id, member_ids in clusters.items()
        ]


def plot(ax, clusters, features):
    clusters = sorted(clusters, key=lambda x: x[0][0])
    for i, (mean, members) in enumerate(clusters):
        color = 'C{}'.format(i % 9)
        ax.scatter(members[:, 0], members[:, 1], c=color, s=1)
        ax.scatter([mean[0]], [mean[1]], c=color, s=100)

    ax.grid()
    ax.set_xlabel(features[0])
    ax.set_ylabel(features[1])


def plot_kmeans(ax, X, clusters, features):
    clusters = [(mean, X[member_ids]) for mean, member_ids in clusters]
    plot(ax, clusters, features)


def plot_actual(ax, df, features):
    clusters = []
    for i, s in enumerate(df.Species.unique()):
        cluster = df[df.Species == s][features].values
        mean = np.mean(cluster, axis=0)
        clusters.append((mean, cluster))
    plot(ax, clusters, features)




if __name__ == '__main__':
    df = pd.read_csv('iris.csv')

    features = [
        'SepalLengthCm',
        # 'PetalLengthCm',
        # 'SepalWidthCm',
        'PetalWidthCm'
    ]

    df_features = df[features]
    k = len(df.Species.unique())
    kms = KMeans(df_features.values, k=k, e=.01)

    df['Label'] = kms.Y

    df.to_csv('out.csv')

    print('features: {}'.format(features))
    print('iterations: {}'.format(kms.its))

    for s in df.Species.unique():
        print('')
        for c in range(k):
            count = len(df[(df.Label == c) & (df.Species == s)])
            print('species = {:20s} label = {}: {:5d}'.format(s, c, count))


    fig = plt.figure(figsize=(10, 4))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    plot_kmeans(ax1, kms.X, kms.clusters, features)
    plot_actual(ax2, df, features)
    ax1.set_title('k-means')
    ax2.set_title('actual')
    fig.show()

