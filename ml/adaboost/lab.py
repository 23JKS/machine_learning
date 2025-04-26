import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

def plot_decision_boundary(model, X, y):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X[y == -1, 0], X[y == -1, 1], label='Class -1', color='red')
    plt.scatter(X[y == 1, 0], X[y == 1, 1], label='Class 1', color='blue')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.show()

# plot_decision_boundary(adaboost, X_train, y_train)
class DecisionStump:
    def __init__(self):
        self.feature_idx = None
        self.threshold = None
        self.alpha = None
        self.polarity = 1

    def predict(self, X):
        n_samples = X.shape[0]
        X_feature = X[:, self.feature_idx]
        predictions = np.ones(n_samples)
        if self.polarity == 1:
            predictions[X_feature < self.threshold] = -1
        else:
            predictions[X_feature >= self.threshold] = -1
        return predictions
class Adaboost:
    def __init__(self, n_estimators):
        self.n_estimators = n_estimators
        self.stumps = []
    def fit(self,X,y):
        #样本数量和特征数量
        n_samples,n_feature=X.shape
        # 初始化权重
        weights=np.ones(n_samples)/n_samples
        for _ in range(self.n_estimators):
            stump=DecisionStump()
            mini_error=float('inf')
            for feature_idx in range(n_feature):
                X_feature=X[:,feature_idx]
                # 所有可能的分界点
                thresholds=np.unique(X_feature)
                for threshold in thresholds:
                    # 两种决策方式
                    for polarity in [1,-1]:
                        pred=np.ones(n_samples)
                        if polarity==1:
                            pred[X_feature<threshold]=-1
                        else:
                            pred[X_feature>=threshold]=-1
                        error=np.sum(weights[pred!=y])
                        # 决策方式取反
                        if(error>0.5):
                            error=1-error
                            polarity*=-1
                        if error <mini_error:
                            # 最优决策树桩
                            mini_error=error
                            stump.polarity=polarity
                            stump.threshold=threshold
                            stump.feature_idx=feature_idx
            stump.alpha=0.5*np.log((1 - mini_error) / mini_error)
            # 更新样本权重，分错的样本权重会变大
            predictions = stump.predict(X)
            weights *= np.exp(-stump.alpha * y * predictions)
            weights /= np.sum(weights)
            self.stumps.append(stump)
            # plot_decision_boundary(self, X_train, y_train)
            # plot_decision_boundary(self, X_test, y_test)

    def predict(self, X):
        stump_preds = np.array([stump.alpha * stump.predict(X) for stump in self.stumps])
        return np.sign(np.sum(stump_preds, axis=0))

def load_data(filename):
    data = []
    with open(filename, 'r') as f:
        for line in f:
            row = line.strip().replace('[', '').replace(']', '').split()
            data.append([float(x) for x in row])
    return np.array(data)

train_data = load_data('train.txt')
test_data = load_data('test.txt')

X_train, y_train = train_data[:, :2], train_data[:, 2]
X_test, y_test = test_data[:, :2], test_data[:, 2]

adaboost = Adaboost(n_estimators=100000)
adaboost.fit(X_train, y_train)

y_train_pred = adaboost.predict(X_train)
y_test_pred = adaboost.predict(X_test)

train_accuracy = np.mean(y_train_pred == y_train)
test_accuracy = np.mean(y_test_pred == y_test)

print(f"训练集准确率: {train_accuracy:.4f}")
print(f"测试集准确率: {test_accuracy:.4f}")
plot_decision_boundary(adaboost, X_train, y_train)
plot_decision_boundary(adaboost, X_test, y_test)