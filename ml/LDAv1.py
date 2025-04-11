import numpy as np
import pandas as pd
from keras.src.metrics.accuracy_metrics import accuracy
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split


data = pd.read_csv('blood_data.txt', header=None, names=['Feature1', 'Feature2', 'Feature3', 'Feature4', 'Label'])
# 划分数据集
X = data[['Feature1', 'Feature2', 'Feature3', 'Feature4']].values  # 特征
y = data['Label'].values  # 标签
test_size = 600
train_size = X.shape[0]- test_size
# 划分训练集和测试集，
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, train_size=train_size, random_state=54, stratify=y)



# 数据预处理
def load_data(X,Y):
    x0=X[Y==0]
    x1=X[Y==1]
    return x0,x1


# 训练集两类分开
x0,x1=load_data(X_train,y_train)

# 对两份x计算协方差矩阵
# 直接调用np中的方法也可以使用上面写的get_cov方法
sigma0=np.cov(x0,rowvar=False)
sigma1=np.cov(x1,rowvar=False)

# 求向量均值
mu0=x0.mean(axis=0)
mu1=x1.mean(axis=0)
# 初始化方向向量
w=np.ones(X_test.shape[1])
# 最小化函数，即求瑞利商
def get_J():
    return np.power(w.dot(mu0)-w.dot(mu1),2)/(w.dot(sigma0).dot(w)+w.dot(sigma1).dot(w))
# 求梯度
def gradient():
    global w
    # 变化幅度，根据实际调整
    upsilon=0.01
    # 梯度
    res=[]
    # 求各变量的偏导
    for i in range(x0.shape[1]):
        l1=get_J()
        w[i]+=upsilon
        l2=get_J()
        w[i]-=upsilon
        res.append((l2-l1)/upsilon)
    # 返回梯度
    return np.array(res)



# 投影可视化
def draw(x0,x1):

    p0 = w.dot(x0.T)
    p1=w.dot(x1.T)
    plt.scatter(p0,np.zeros(len(p0)),c='blue')
    plt.scatter(p1, np.zeros(len(p1)), c='red')

    plt.scatter(p0.mean(),1,c='blue')
    plt.scatter(p1.mean(), 1, c='red')
    plt.show()

# 预测
def predict(x):
    p=w.dot(x.T)
    p0 = w.dot(x0.T)
    p1 = w.dot(x1.T)
    d0=np.power(p-p0.mean(),2)
    d1 = np.power(p - p1.mean(), 2)
    pred=1 if d0>d1 else 0
    # print(pred)
    return pred


def cal_ac(X_test):
    accura=0;
    for i in range(X_test.shape[0]):

        if predict(X_test[i,:])==y_test[i]:
            accura+=1
    return accura/X_test.shape[0]


# 训练
# 迭代次数
# print(x1.shape[0]+x0.shape[0])
iter = 64380
# 学习率
learn_rate = 0.1

for i in range(iter):
    # 最大化瑞利商 梯度上升
    w += gradient() * learn_rate

    # 打印结果以观察训练情况
    if i % 10 == 0:
        print(get_J(), w)


# 测试集投影后可视化
draw(X_test[y_test==0],X_test[y_test==1])
print("方向向量w",w)
print("准确率",cal_ac(X_test))