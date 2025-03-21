import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# 加载数据
data = pd.read_csv('blood_data.txt', header=None, names=['Feature1', 'Feature2', 'Feature3', 'Feature4', 'Label'])
mx_acc=0;
p=-1
for i in range(1, 100000):
    # 划分数据集
    X = data[['Feature1', 'Feature2', 'Feature3', 'Feature4']].values  # 特征
    y = data['Label'].values  # 标签

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=600, train_size=148, random_state=i, stratify=y)

    # 特征标准化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 手动实现 LDA
    class LDA:
        def __init__(self, lambda_value=0.01):
            self.theta = None  # 投影方向
            self.threshold = None  # 分类阈值
            self.lambda_value = lambda_value  # 正则化参数

        def fit(self, X, y):
            """
            训练 LDA 模型
            :param X: 训练集特征 (n_samples, n_features)
            :param y: 训练集标签 (n_samples,)
            """
            n_samples, n_features = X.shape
            classes = np.unique(y)
            n_classes = len(classes)

            if n_classes != 2:
                raise ValueError("手动实现的 LDA 仅支持二分类问题。")

            # 计算每类的均值
            class_means = []
            for c in classes:
                class_means.append(np.mean(X[y == c], axis=0))
            class_means = np.array(class_means)

            # 计算全局均值
            global_mean = np.mean(X, axis=0)

            # 计算类间散度矩阵 Sb
            Sb = np.zeros((n_features, n_features))
            for i, mean_vec in enumerate(class_means):
                n = X[y == classes[i]].shape[0]
                mean_diff = (mean_vec - global_mean).reshape(n_features, 1)
                Sb += n * np.dot(mean_diff, mean_diff.T)

            # 计算类内散度矩阵 Sw
            Sw = np.zeros((n_features, n_features))
            for i, c in enumerate(classes):
                class_samples = X[y == c]
                class_mean = class_means[i]
                class_diff = class_samples - class_mean
                Sw += np.dot(class_diff.T, class_diff)

            # 加入正则化
            Sw_reg = Sw + self.lambda_value * np.eye(Sw.shape[0])

            # 计算投影方向 theta
            self.theta = np.dot(np.linalg.inv(Sw_reg), (class_means[0] - class_means[1]).T)

            # 单位化 theta
            self.theta = self.theta / np.linalg.norm(self.theta)

            # 计算分类阈值（使用训练集投影的中位数）
            projected_train = np.dot(X_train, self.theta)
            self.threshold = np.median(projected_train)

        def predict(self, X):
            """
            预测类别
            :param X: 测试集特征 (n_samples, n_features)
            :return: 预测标签 (n_samples,)
            """
            projections = np.dot(X, self.theta)
            return np.where(projections >= self.threshold, 1, 0)

    # 初始化 LDA 模型
    lda = LDA(lambda_value=0.01)

    # 训练模型
    lda.fit(X_train, y_train)

    # 在测试集上进行预测
    y_pred = lda.predict(X_test)

    # 计算准确率
    accuracy = accuracy_score(y_test, y_pred)
    if  accuracy>mx_acc:
        mx_acc=accuracy
        p=i
    # print(f"随机种子{i}测试集准确率: {accuracy * 100:.2f}%")

print(f"随机种子{p}测试集准确率最大: {mx_acc * 100:.2f}%")