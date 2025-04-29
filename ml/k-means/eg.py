import matplotlib.pyplot as plt
import numpy as np
import math

from sqlalchemy.sql.functions import random
from statsmodels.stats.power import normal_sample_size_one_tail
from sklearn.datasets import make_blobs

def load_data(filename):
    data = []
    with open(filename, 'r') as f:
        for raw in f:
            raw = raw.replace('[', '').replace(']', '').split()
            data.append([float(x) for x in raw])
    return np.array(data)
#
# data = load_data('data.txt')
# X = data[:, :2]
X,true_labels=make_blobs(n_samples=1000,n_features=2,centers=3,cluster_std=1.0,random_state=42)
plt.scatter(X[:,0],X[:,1],c='r')
plt.show()
def k_means(X, cluster=3):
    n = X.shape[0]  # 样本数
    indices = np.random.choice(n, size=cluster, replace=False)
    c = X[indices, :]  # 初始化簇中心
    cluster_assignment = np.zeros(n, dtype=int)  # 记录每个样本的簇标签

    while True:
        # 分配样本到最近的簇
        for j in range(n):
            # 修正距离计算：使用欧氏距离 (c[i][0] - X[j][0])^2 + (c[i][1] - X[j][1])^2
            distances = [math.sqrt((c[i][0] - X[j][0])**2 + (c[i][1] - X[j][1])**2) for i in range(cluster)]
            cluster_assignment[j] = np.argmin(distances)

        # 更新簇中心
        new_c = np.zeros_like(c)
        for i in range(cluster):
            points_in_cluster = X[cluster_assignment == i]
            if len(points_in_cluster) > 0:
                new_c[i] = np.mean(points_in_cluster, axis=0)
            else:
                new_c[i] = c[i]  # 如果簇为空，保持原中心

        # 检查是否收敛（新旧中心差异小于阈值）
        if np.allclose(c, new_c, atol=1e-5):
            break
        c = new_c

    return c, cluster_assignment

# 运行K-Means
centers, labels = k_means(X, 3)

# 可视化聚类结果
plt.figure(figsize=(8, 6))
colors = ['r', 'g', 'b', 'c', 'm', 'y']  # 定义簇的颜色
for i in range(len(centers)):
    # 绘制每个簇的点
    cluster_points = X[labels == i]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], c=colors[i], label=f'Cluster {i+1}', alpha=0.6)
    # 绘制簇中心
    plt.scatter(centers[i, 0], centers[i, 1], c=colors[i], marker='x', s=200, linewidths=3)

plt.title('K-Means Clustering Results')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True)
plt.show()