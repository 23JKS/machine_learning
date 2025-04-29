import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn.datasets import make_blobs

def load_data(filename):
    data=[]
    with open(filename,'r') as f:
        for raw in f:
            raw=raw.replace('[','').replace(']','').split()
            data.append([float(x) for x in raw])
    return np.array(data)

X,true_labels=make_blobs(n_samples=1000,n_features=2,centers=3,cluster_std=1,random_state=42)
# print(X)
plt.scatter(X[:,0],X[:,1],c='r')
plt.show()
def k_means(X,cluster=3):
    # 样本点数
    n=X.shape[0]
    # 随机选择cluster个簇中心
    indices = np.random.choice(n, size=cluster, replace=False)
    # print(indices)
    c=X[indices,:]
    #
    flag=1
    cluster_point=[[] for _ in range(0,cluster)]
    while flag:
        flag=0
        part=[[] for _ in range(0,cluster) ]
        for j in range(0,n):
            dist=[]
            for i in range(0,cluster):
                d=math.sqrt((c[i][0]-X[j][0])**2+(c[i][1]-X[j][1])**2)
                dist.append(d)
            dist=np.array(dist)
            dis=np.argsort(dist)
            p=dis[0]
            # 分到最近的簇中心点所在的簇
            part[p].append(j)
        # 更新簇中心
        nc=[[] for _ in range(0,cluster) ]
        for j in range(0,cluster):
            nc[j]=X[part[j],:].mean(axis=0)
            # print(nc[j])
            if abs(nc[j][0]-c[j][0])>1e-9 or abs(nc[j][1]-c[j][1])>1e-9:
                c[j][0]=nc[j][0]
                c[j][1]=nc[j][1]
                # 标记是否有更新操作
                flag=1
        cluster_point=part
    # for j in range(0, n):
    #     dist = []
    #     for i in range(0, cluster):
    #         d = math.sqrt((c[i][0] - X[j][0]) ** 2 + (c[i][1] - X[j][1]) ** 2)
    #         dist.append(d)
    #
    #     dist = np.array(dist)
    #     dis = np.argsort(dist)
    #     p = dis[0]
    #
    #     cluster_point[p].append(j)
    # 可视化聚类结果
    plt.figure(figsize=(8, 6))
    colors = ['r', 'g', 'b', 'c', 'm', 'y']  # 定义簇的颜色
    for i in range(0,cluster):

        plt.scatter(X[cluster_point[i],0], X[cluster_point[i],1], c=colors[i], label=f'Cluster {i + 1}', alpha=0.6)
        # 绘制簇中心
        plt.scatter(c[i, 0], c[i, 1], c=colors[i], marker='x', s=200, linewidths=3)

    plt.title('K-Means Clustering Results')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.grid(True)
    plt.show()


k_means(X,3)



