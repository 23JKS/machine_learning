# from audioop import error

import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler  # 新增特征归一化
'''
手动实现的knn算法
'''
# 加载数据
iris = load_iris()
X, y = shuffle(iris.data, iris.target, random_state=13)

# 特征归一化 (关键改进)
scaler = StandardScaler()
X = scaler.fit_transform(X)  # 统一归一化

# 划分训练集和测试集
offset = int(X.shape[0] * 0.7)
X_train, y_train = X[:offset], y[:offset]
X_test, y_test = X[offset:], y[offset:]

# 转换标签形状
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

def compute_distances(X, X_train):
    num_test = X.shape[0]
    num_train = X_train.shape[0]
    M = np.dot(X, X_train.T)
    te2 = np.square(X).sum(axis=1)
    tr2 = np.square(X_train).sum(axis=1)
    # 修正距离计算，避免负数
    te2_reshaped = te2.reshape(-1, 1)  # 替换 np.matrix
    dists_squared = -2 * M + tr2 + te2_reshaped
    dists_squared = np.maximum(dists_squared, 0)  # 确保非负
    return np.sqrt(dists_squared)

def predict_labels(y_train, dists, k=1):
    num_test = dists.shape[0]
    y_pred = np.zeros(num_test)
    for i in range(num_test):
        closest_indices = np.argsort(dists[i, :])[:k]
        closest_y = y_train[closest_indices].flatten()
        c = Counter(closest_y)
        # 处理平票情况：随机选择众数中的一个
        most_common = c.most_common()
        max_count = most_common[0][1]
        candidates = [num for num, count in most_common if count == max_count]
        y_pred[i] = np.random.choice(candidates)
    return y_pred

# 计算距离矩阵
dists = compute_distances(X_test, X_train)
plt.imshow(dists, interpolation='none')
plt.show()

# 测试不同 k 值
k_values = range(1, 20)  # 扩大 k 范围
accuracies = []
for k in k_values:
    y_test_pred = predict_labels(y_train, dists, k=k)
    num_correct = np.sum(y_test_pred.reshape(-1, 1) == y_test)
    accuracy = num_correct / y_test.shape[0]
    accuracies.append(accuracy)
    print(f'k={k}: Accuracy={accuracy:.4f}')

# 可视化错误率
plt.figure(figsize=(10, 6))
error=[]
for i in accuracies:
  error.append(1-i)
# print(error)
plt.plot(k_values, error, 'b-o')
plt.xlabel('k')
plt.ylabel('Error_rate')
plt.title('KNN Accuracy with Feature Scaling')
plt.grid(True)
plt.show()