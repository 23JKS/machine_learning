import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score

# 加载数据
data = pd.read_csv('blood_data.txt', header=None, names=['Feature1', 'Feature2', 'Feature3', 'Feature4', 'Label'])

# 划分数据集
X = data[['Feature1', 'Feature2', 'Feature3', 'Feature4']]  # 特征
y = data['Label']  # 标签

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=600, train_size=148, random_state=42, stratify=y)

# 初始化 LDA 模型
lda = LinearDiscriminantAnalysis()

# 训练模型
lda.fit(X_train, y_train)

# 在测试集上进行预测
y_pred = lda.predict(X_test)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print(f"测试集准确率: {accuracy * 100:.2f}%")