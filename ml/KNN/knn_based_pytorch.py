import torch
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler

# 加载数据
iris = load_iris()
X, y = shuffle(iris.data, iris.target, random_state=13)

# 特征归一化
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 转换为PyTorch张量
X = torch.FloatTensor(X)
y = torch.LongTensor(y)

# 划分训练集和测试集
offset = int(X.shape[0] * 0.7)
X_train, y_train = X[:offset], y[:offset]
X_test, y_test = X[offset:], y[offset:]


def compute_distances(X, X_train):
    """PyTorch向量化距离计算"""
    # 计算点积矩阵 (test x train)
    M = torch.mm(X, X_train.T)

    # 计算平方和
    te2 = torch.sum(X ** 2, dim=1).view(-1, 1)  # test samples (n_test, 1)
    tr2 = torch.sum(X_train ** 2, dim=1)  # train samples (n_train,)

    # 欧式距离平方
    dists_squared = -2 * M + tr2 + te2
    dists_squared = torch.clamp(dists_squared, min=0)  # 避免负数
    return torch.sqrt(dists_squared)


def predict_labels(y_train, dists, k=1):
    """PyTorch KNN预测"""
    _, topk_indices = torch.topk(dists, k=k, dim=1, largest=False)
    closest_y = y_train[topk_indices]

    # 统计最近k个样本的类别
    y_pred = torch.zeros_like(y_test)
    for i in range(closest_y.shape[0]):
        counts = torch.bincount(closest_y[i], minlength=3)
        y_pred[i] = torch.argmax(counts)
    return y_pred


# 计算距离矩阵
dists = compute_distances(X_test, X_train)

# 可视化距离矩阵
plt.imshow(dists.numpy(), interpolation='none')
plt.colorbar()
plt.title("Pairwise Distance Matrix")
plt.show()

# 测试不同k值
k_values = range(1, 20)
accuracies = []
error_rates = []

for k in k_values:
    y_pred = predict_labels(y_train, dists, k=k)
    correct = torch.sum(y_pred == y_test).item()
    accuracy = correct / y_test.shape[0]
    accuracies.append(accuracy)
    error_rates.append(1 - accuracy)
    print(f'k={k}: Accuracy={accuracy:.4f}, Error Rate={1 - accuracy:.4f}')

# 可视化错误率
plt.figure(figsize=(10, 6))
plt.plot(k_values, error_rates, 'b-o')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Error Rate')
plt.title('KNN Error Rate vs. k Value (PyTorch)')
plt.xticks(k_values)
plt.grid(True)
plt.show()