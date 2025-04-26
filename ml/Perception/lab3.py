import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 读取数据
data = pd.read_csv('perceptron_data.txt', sep='\t', header=None, names=['x1', 'x2', 'label'])

# 将标签0转换为-1
data['label'] = data['label'].apply(lambda x: -1 if x == 0 else 1)

# 提取特征和标签
X = data[['x1', 'x2']].values
y = data['label'].values

# 初始化权重和偏置
W = np.zeros(2)  # 权重
θ = 0  # 偏置
learning_rate = 0.1


# 定义sign激活函数
def sign(x):
    return 1 if x >= 0 else -1

# 计算决策边界
def decision_boundary(x1, W, θ):
    return (-θ - W[0] * x1) / W[1]
# 训练感知机
def train_perceptron(X, y, W, θ, learning_rate, max_epochs=100):
    epoch = 0
    updated = True

    while updated and epoch < max_epochs:
        updated = False
        print(f"\nEpoch {epoch + 1}:")

        for i in range(len(X)):
            xi = X[i]
            yi = y[i]

            # 计算预测值
            prediction = sign(np.dot(W, xi) + θ)

            # 如果分类错误，更新权重和偏置
            if prediction != yi:
                W += learning_rate * yi * xi
                θ += learning_rate * yi
                updated = True

                # 打印更新后的参数
                print(f"After sample {i + 1}: W = [{W[0]:.6f}, {W[1]:.6f}], θ = {θ:.6f}")

        epoch += 1
        # 绘制数据点
        plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='blue', label='Class 1')
        plt.scatter(X[y == -1][:, 0], X[y == -1][:, 1], color='red', label='Class -1')

        # 绘制决策边界
        x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        x1_values = np.linspace(x1_min, x1_max, 100)
        x2_values = decision_boundary(x1_values, W, θ)
        plt.plot(x1_values, x2_values, 'g--', label='Decision Boundary')

        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.title('Perceptron Decision Boundary')
        plt.legend()
        plt.grid(True)
        plt.show()
    return W, θ


# 训练模型
W_final, θ_final = train_perceptron(X, y, W, θ, learning_rate)

# 打印最终结果
print("\nFinal Parameters:")
print(f"W = [{W_final[0]:.6f}, {W_final[1]:.6f}]")
print(f"θ = {θ_final:.6f}")





# 可视化结果
plt.figure(figsize=(10, 6))

# 绘制数据点
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='blue', label='Class 1')
plt.scatter(X[y == -1][:, 0], X[y == -1][:, 1], color='red', label='Class -1')

# 绘制决策边界
x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
x1_values = np.linspace(x1_min, x1_max, 100)
x2_values = decision_boundary(x1_values, W_final, θ_final)
plt.plot(x1_values, x2_values, 'g--', label='Decision Boundary')

plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Perceptron Decision Boundary')
plt.legend()
plt.grid(True)
plt.show()

# 输出决策边界方程
print("\nDecision Boundary Equation:")
print(f"{W_final[0]:.6f}x1 + {W_final[1]:.6f}x2 + {θ_final:.6f} = 0")