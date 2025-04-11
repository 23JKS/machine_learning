import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 加载数据
data = pd.read_csv('blood_data.txt', header=None,
                   names=['Feature1', 'Feature2', 'Feature3', 'Feature4', 'Label'])
X = data[['Feature1', 'Feature2', 'Feature3', 'Feature4']].values
y = data['Label'].values

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=600, train_size=148, random_state=54, stratify=y)

# 数据标准化（PCA前必须做标准化）
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# PCA降维（保留95%的方差）
pca = PCA(n_components=0.95)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

print(f"原始特征数: {X.shape[1]}，降维后特征数: {X_train_pca.shape[1]}")


# 更新后的LDA实现（使用PCA降维后的数据）
def lda_with_pca(X_train, y_train, X_test, y_test):
    # 分离两类数据
    x0 = X_train[y_train == 0]
    x1 = X_train[y_train == 1]

    # 计算均值和协方差
    mu0, mu1 = x0.mean(axis=0), x1.mean(axis=0)
    sigma0 = np.cov(x0, rowvar=False)
    sigma1 = np.cov(x1, rowvar=False)

    # 计算合并的类内散度矩阵
    S_w = (len(x0) * sigma0 + len(x1) * sigma1) / len(X_train)

    # 计算LDA方向（解析解）
    w = np.linalg.pinv(S_w).dot(mu1 - mu0)
    w = w / np.linalg.norm(w)  # 归一化

    # 计算投影
    def project(X):
        return X.dot(w)

    # 预测函数
    def predict(X):
        projections = project(X)
        threshold = (project(mu0.reshape(1, -1)) + project(mu1.reshape(1, -1))) / 2
        return (projections > threshold).astype(int)

    # 评估
    y_pred = predict(X_test)
    accuracy = np.mean(y_pred == y_test)

    # 可视化
    def plot_projection():
        plt.figure(figsize=(10, 2))
        proj_test = project(X_test)
        plt.scatter(proj_test[y_test == 0], np.zeros_like(proj_test[y_test == 0]),
                    alpha=0.5, label='Class 0')
        plt.scatter(proj_test[y_test == 1], np.zeros_like(proj_test[y_test == 1]),
                    alpha=0.5, label='Class 1')
        plt.axvline(x=threshold, color='k', linestyle='--')
        plt.legend()
        plt.title('LDA Projection after PCA')
        plt.show()

    threshold = (project(mu0.reshape(1, -1)) + project(mu1.reshape(1, -1))) / 2
    plot_projection()

    return w, accuracy


# 运行LDA
w_pca, acc_pca = lda_with_pca(X_train_pca, y_train, X_test_pca, y_test)
print("PCA降维后的方向向量:", w_pca)
print("PCA降维后的准确率:", acc_pca)


# 对比原始LDA（不使用PCA）
def original_lda(X_train, y_train, X_test, y_test):
    x0, x1 = X_train[y_train == 0], X_train[y_train == 1]
    mu0, mu1 = x0.mean(axis=0), x1.mean(axis=0)
    sigma0, sigma1 = np.cov(x0, rowvar=False), np.cov(x1, rowvar=False)
    S_w = (len(x0) * sigma0 + len(x1) * sigma1) / len(X_train)
    w = np.linalg.pinv(S_w).dot(mu1 - mu0)
    w = w / np.linalg.norm(w)

    projections = X_test.dot(w)
    threshold = (mu0.dot(w) + mu1.dot(w)) / 2
    y_pred = (projections > threshold).astype(int)

    return np.mean(y_pred == y_test)


orig_acc = original_lda(X_train_scaled, y_train, X_test_scaled, y_test)
print("\n原始数据LDA准确率:", orig_acc)
print("PCA+LDA准确率:", acc_pca)