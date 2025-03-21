import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
# 加载数据
data = pd.read_csv('blood_data.txt', header=None, names=['Feature1', 'Feature2', 'Feature3', 'Feature4', 'Label'])

# 划分数据集
X = data[['Feature1', 'Feature2', 'Feature3', 'Feature4']].values  # 特征
y = data['Label'].values  # 标签
test_size=600
train_size=748-test_size

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, train_size=train_size, random_state=54, stratify=y)
# 特征标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# data.to_csv("ss");
# 手动实现 LDA
class LDA:
    def __init__(self):
        self.theta = None  # 投影方向
        self.threshold = None  # 分类阈值

    def fit(self, X, y):
        """
        训练 LDA 模型
        :param X: 训练集特征 (n_samples, n_features)
        :param y: 训练集标签 (n_samples,)
        """
        n_samples, n_features = X.shape
        classes = np.unique(y)
        n_classes = len(classes)

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

        # 计算投影方向 theta
        self.theta = np.dot(np.linalg.inv(Sw), (class_means[0] - class_means[1]).T)

        # 计算分类阈值
        projected_means = np.dot(class_means, self.theta)
        self.threshold = np.mean(projected_means)


    def predict(self, X):
        """
        预测类别
        :param X: 测试集特征 (n_samples, n_features)
        :return: 预测标签 (n_samples,)
        """
        projections = np.dot(X, self.theta)
        return np.where(projections >= self.threshold, 1, 0)

# 初始化 LDA 模型
lda = LDA()

# 训练模型
lda.fit(X_train, y_train)

# 在测试集上进行预测
y_pred = lda.predict(X_test)
print("投影方向:",lda.theta)
# 计算准确率
accuracy = accuracy_score(y_test, y_pred)

print(f"测试集准确率: {accuracy * 100:.2f}%")

# 可视化投影分布
def plot_projections(X, y, w,threshold ,title):
    projections = X @ w
    plt.figure(figsize=(10, 6))
    plt.hist(projections[y == 0], bins=20, alpha=0.5, label='Class 0', color='blue')
    plt.hist(projections[y == 1], bins=20, alpha=0.5, label='Class 1', color='red')
    plt.axvline(threshold, color='black', linestyle='--', label='Threshold')
    plt.title(title)
    plt.xlabel("Projection Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()

plot_projections(X_train, y_train, lda.theta, lda.threshold,"Training Set Projections")
plot_projections(X_test, y_test, lda.theta,lda.threshold, "Test Set Projections")