import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data=[]
# 读取数据
with  open ("data.txt") as file:
    for line in file:
        line=line.replace('[','').replace(']','').split()
        print(line)
        data.append([float(x) for x in line])
data=np.array(data);
# cov(x1,x2)=E((x1-mean(x1))*(x2-mean(x2)))
def cal_cov(x1,x2):
    sigma1=np.mean(x1,0)
    sigma2 = np.mean(x2, 0)
    x=x1*x2
    cov=np.mean(x,0)-sigma1*sigma2
    return cov

x1=data[:,0]
x2=data[:,1]
# 样本进行去中心化
x1=x1-np.mean(x1)
x2=x2-np.mean(x2)
# 协方差矩阵
cov_m=np.zeros((2,2))
cov_m[0,0]=np.var(x1)
cov_m[0,1]=cal_cov(x1,x2)
cov_m[1,0]=cov_m[0,1]
cov_m[1,1]=np.var(x2)
print(cov_m)

# numpy包计算协方差来验证手动计算结果
cov12=np.cov(x1,x2)

print(cov12)

# 对协方差矩阵做特征值分解
eigen_values, eigen_vectors = np.linalg.eig(cov_m)

# 按特征值从大到小排序特征向量
sorted_indices = np.argsort(eigen_values)[::-1]
eigen_values = eigen_values[sorted_indices]
eigen_vectors = eigen_vectors[:, sorted_indices]
eigen_vectors=eigen_vectors[:, 0]
print("特征值:", eigen_values)
print("特征向量矩阵（每列为一个主成分）:\n", eigen_vectors)

# 将去中心化的数据投影到主成分上
data_centered = np.column_stack((x1, x2))  # 已去中心化的数据
pca_data = data_centered.dot(eigen_vectors)

print("\n投影后的PCA数据:")
print(pca_data)


# --- 可视化对比 ---
plt.figure(figsize=(12, 5))

# 原始数据（去中心化后的二维分布）
plt.subplot(1, 2, 1)
plt.scatter(data_centered[:, 0], data_centered[:, 1], alpha=0.5)
plt.title("Original Data (Centered)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")



# --- 计算方差解释率 ---
total_variance = np.sum(eigen_values)
explained_variance_ratio = eigen_values[0] / total_variance
print(f"主成分1解释的方差比例: {explained_variance_ratio:.2%}")