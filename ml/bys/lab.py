# 导入数据集
import numpy as np
import pandas as pd

data=pd.read_csv('data.txt',names=['x1','x2','label'])


X1=np.array(data['x1'])
X2=np.array(data['x2'])
y=np.array(data['label'])

cnt1=0
for i in range(0,len(y)):
    if y[i]==1:
        cnt1+=1
p1=cnt1/len(y)
cnt0=len(y)-cnt1
p0=1-p1


def predict(x1,x2):
    p=[0,0]
    # 计算后验概率
    for c in [-1,1]:
        cnt=0
        for i in range(0,len(X1)):
            if X1[i]==x1 and c==y[i]:
                cnt+=1
        # p(x1| y)
        if c==1:
            px1_c=cnt/cnt1
        else:
            px1_c=cnt/cnt0
        cnt=0
        for i in range(0,len(X2)):
            if X2[i]==x2 and c==y[i]:
                cnt+=1
        # p(x2 | y)
        if c == 1:
            px2_c = cnt / cnt1
        else:
            px2_c = cnt / cnt0
        # 计算p(x1,x2|y)*p(y)
        if c==-1:
            p[0]=px1_c*px2_c*p0
        else:
            p[1]=px1_c*px2_c*p1
    # 返回使得后验概率最大的类别
    if p[0]>p[1]:
        return -1
    else:
        return 1
x=[2,'S']
pred=predict(x[0],x[1])
print("该样本属于:",pred,)
