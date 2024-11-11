import numpy as np
import matplotlib.pyplot as plt

X = np.array([
    [23, 2, 11, 26],
    [24, 3, 13, 22],
    [22, 4, 19, 15],
    [28, 5, 16, 17],
    [26, 6, 12, 27],
    [23, 8, 11, 23],
    [21, 9, 17, 13],
    [29, 10, 18, 18],
    [22, 10, 13, 29],
    [27, 11, 11, 25],
    [25, 13, 15, 14],
    [28, 14, 16, 21],
    [26, 15, 19, 28],
    [23, 16, 17, 17],
    [21, 17, 12, 24],
    [29, 24, 13, 12],
    [23, 25, 11, 6],
    [24, 26, 15, 9],
    [22, 27, 16, 4],
    [28, 27, 19, 14],
    [26, 29, 17, 12],
    [23, 30, 13, 8],
    [21, 30, 19, 15],
    [29, 32, 16, 2],
    [22, 32, 12, 11],
    [27, 33, 11, 6],
    [25, 34, 17, 13],
    [28, 35, 18, 3],
    [26, 36, 11, 10],
    [23, 37, 14, 6],
    [21, 20, 17, 44],
    [29, 21, 13, 48],
    [25, 22, 19, 40],
    [23, 23, 16, 45],
    [24, 23, 12, 51],
    [22, 25, 11, 53],
    [28, 26, 17, 39],
    [26, 26, 18, 44],
    [23, 26, 11, 50],
    [21, 27, 14, 36],
    [29, 29, 15, 46],
    [22, 30, 13, 41],
    [27, 30, 19, 52],
    [25, 32, 17, 43],
    [28, 32, 12, 48],
    [29, 7, 11, 35],
    [22, 4, 19, 36],
    [27, 7, 13, 37],
    [25, 10, 10, 37],
    [28, 6, 11, 39],
    [26, 9, 18, 39],
    [23, 11, 15, 40],
    [21, 5, 11, 41],
    [29, 7, 13, 42],
    [25, 13, 10, 43],
    [23, 10, 17, 44],
    [24, 8, 11, 45],
    [22, 12, 18, 46],
    [28, 9, 15, 48],
    [26, 11, 11, 50],
])


def manhattan_distance(x1, x2):
    return np.sum(np.abs(x1 - x2))

def initialize_centroids(X, k):
    n_samples, n_features = X.shape
    centroids = np.zeros((k, n_features))
    for i in range(k):
        centroid = X[np.random.choice(range(n_samples))]
        print('Выбранный объект как центр кластера №', i+1 ,centroid)
        centroids[i] = centroid
    return centroids

def assign_clusters(X, centroids):
    n_samples = X.shape[0]
    clusters = np.zeros(n_samples)
    for i in range(n_samples):
        distances = np.zeros(centroids.shape[0])
        for j in range(centroids.shape[0]):
            distances[j] = manhattan_distance(X[i], centroids[j])
        cluster = np.argmin(distances)
        clusters[i] = cluster
    return clusters

def update_centroids(X, clusters, k):
    n_features = X.shape[1]
    centroids = np.zeros((k, n_features))
    for i in range(k):
        centroid = X[clusters == i].mean(axis=0)
        centroids[i] = centroid
    return centroids

def plot_clusters(X, clusters, centroids):
    colors = ['r', 'g', 'y', 'b']
    markers = ['P', '^', 'p', 's']
    for i in range(len(colors)):
        plt.scatter(X[clusters == i, 0], X[clusters == i, 1], c=colors[i], s=30, marker=markers[i])
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', c='black', s=100)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Clusters')
    plt.show()

def k_means(X, k, max_iters=30):
    global clusters
    centroids = initialize_centroids(X, k)
    for i in range(max_iters):
        clusters = assign_clusters(X, centroids)
        old_centroids = centroids
        centroids = update_centroids(X, clusters, k)
        if np.all(old_centroids == centroids):
            break
    plot_clusters(X, clusters, centroids)
    return clusters, centroids

n_samples = 60
n_features = 4
objects = np.array([])

print('Выберите 2 признака, которые будут использованы (через enter)(1-4)')
feature1 = int(input()) - 1
feature2 = int(input()) - 1

if feature1 < 0 or feature1 > 3 or feature2 < 0 or feature2 > 3:
    print('Ошибка: номер признака должен быть между 1 и 4')
else:
    k = 4
    clusters, centroids = k_means(X[:, [feature1, feature2]], k)