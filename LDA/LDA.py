import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 加载数据
data = pd.read_csv('blood_data.txt', header=None, names=['Feature1', 'Feature2', 'Feature3', 'Feature4', 'Label'])
data = data.to_numpy()  # 转换为 NumPy 数组


# 定义绘图函数
def draw1(data, theta, num, colors):
    x = [[] for _ in range(num)]
    for ele in data:
        x[int(ele[-1])].append(ele[:-1])  # 根据类别标签分组

    # 绘制原始数据（只绘制前两个特征）
    for i in range(num):
        x[i] = np.array(x[i])
        plt.scatter(x[i][:, 0], x[i][:, 1], color=colors[i], label=f'Class {i}')

    # 绘制映射直线（只绘制前两个特征）
    plt.plot([0, theta[0] * 15], [0, theta[1] * 15], label='Decision Boundary')

    # 绘制映射到直线上的点（只绘制前两个特征）
    for i in range(num):
        for ele in x[i]:
            ta = theta * np.dot(ele[:2], theta[:2])  # 只使用前两个特征
            plt.plot([ele[0], ta[0]], [ele[1], ta[1]], color=colors[i], linestyle="--")
            plt.scatter(ta[0], ta[1], color=colors[i])

    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('LDA Classification')
    plt.legend()
    plt.show()


# 定义 LDA 计算函数
def c2(data, num):
    n = data.shape[1] - 1  # 特征数量
    x = [[] for _ in range(num)]  # 存储每类的特征
    u = [[] for _ in range(num)]  # 存储每类的均值
    sw = np.zeros([n, n])  # 初始化散度矩阵

    # 按类别分组
    for ele in data:
        x[int(ele[-1])].append(ele[:-1])

    # 计算每类的均值
    for i in range(num):
        x[i] = np.array(x[i])
        u[i] = np.mean(x[i], axis=0)
    print("1. 计算每类的均值：\n", u)

    # 去中心化并计算散度矩阵 Sw
    for i in range(num):
        x[i] = x[i] - u[i]
        sw += np.dot(x[i].T, x[i])
    print("2. 去中心化后的数据：\n", x)
    print("3. 计算散度矩阵 Sw：\n", sw)

    # 计算 theta
    theta = np.dot(np.linalg.inv(sw), (u[0] - u[1]).T)

    # 单位化
    theta = theta / np.linalg.norm(theta)
    return theta


# 定义颜色
colors = ['red', 'green']

# 计算 theta
theta = c2(data, 2)
print("4. 计算直线向量 theta：\n", theta)

# 绘制结果（只使用前两个特征）
draw1(data, theta, 2, colors)