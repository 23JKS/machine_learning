import numpy as np
from collections import Counter

class Node:
    def __init__(self, feature=None, value=None, branches=None):
        self.feature = feature  # 分裂特征
        self.value = value  # 叶节点的值（类别）
        self.branches = branches if branches is not None else {}  # 分支

    def is_leaf(self):
        return self.value is not None

def entropy(y):
    counts = Counter(y)
    probabilities = [count / len(y) for count in counts.values()]
    return -sum(p * np.log2(p) for p in probabilities)

def information_gain(X_column, y, threshold):
    parent_entropy = entropy(y)
    left_indices = X_column <= threshold
    right_indices = X_column > threshold
    left_y, right_y = y[left_indices], y[right_indices]

    if len(left_y) == 0 or len(right_y) == 0:
        return 0

    n = len(y)
    n_left, n_right = len(left_y), len(right_y)
    child_entropy = (n_left / n) * entropy(left_y) + (n_right / n) * entropy(right_y)
    return parent_entropy - child_entropy

def find_best_split(X, y):
    best_gain = 0
    best_feature = None
    best_threshold = None

    for feature_idx in range(X.shape[1]):
        X_column = X[:, feature_idx]
        thresholds = set(X_column)

        for threshold in thresholds:
            gain = information_gain(X_column, y, threshold)
            if gain > best_gain:
                best_gain = gain
                best_feature = feature_idx
                best_threshold = threshold

    return best_feature, best_threshold

def tree_generate(D, A, max_depth=5, depth=0):
    y = D[:, -1]
    if len(set(y)) == 1:
        return Node(value=y[0])


    if len(A) == 0 or all(np.all(D[:, a] == D[0, a]) for a in A):
        majority_class = Counter(y).most_common(1)[0][0]
        return Node(value=majority_class)

    best_feature, best_threshold = find_best_split(D[:, A], y)
    if best_feature is None:
        majority_class = Counter(y).most_common(1)[0][0]
        return Node(value=majority_class)

    feature_idx_in_A = A[best_feature]
    node = Node(feature=feature_idx_in_A)

    for value in set(D[:, feature_idx_in_A]):
        D_v = D[D[:, feature_idx_in_A] == value]
        if len(D_v) == 0:
            majority_class = Counter(y).most_common(1)[0][0]
            node.branches[value] = Node(value=majority_class)
        else:
            node.branches[value] = tree_generate(D_v, [a for a in A if a != feature_idx_in_A], max_depth, depth + 1)

    return node

def predict_sample(tree, x):
    if tree.is_leaf():
        return tree.value

    feature_value = x[tree.feature]
    if feature_value in tree.branches:
        return predict_sample(tree.branches[feature_value], x)
    else:
        return None

def predict(tree, X):
    return [predict_sample(tree, x) for x in X]

# 数据集
data = np.array([
    [1, 1, 1, 1, 3],
    [1, 1, 1, 2, 2],
    [1, 1, 2, 1, 3],
    [1, 1, 2, 2, 1],
    [1, 2, 1, 1, 3],
    [1, 2, 1, 2, 2],
    [1, 2, 2, 1, 3],
    [1, 2, 2, 2, 1],
    [2, 1, 1, 1, 3],
    [2, 1, 1, 2, 2],
    [2, 1, 2, 1, 3],
    [2, 1, 2, 2, 1],
    [2, 2, 1, 1, 3],
    [2, 2, 1, 2, 2],
    [2, 2, 2, 1, 3],
    [2, 2, 2, 2, 3],
    [3, 1, 1, 1, 3],
    [3, 1, 1, 2, 3],
    [3, 1, 2, 1, 3],
    [3, 1, 2, 2, 1],
    [3, 2, 1, 1, 3],
    [3, 2, 1, 2, 2],
    [3, 2, 2, 1, 3],
    [3, 2, 2, 2, 3]
])

# 特征和目标变量
X = data[:, :-1]
y = data[:, -1]

# 构建决策树
print(X.shape[1])
A = list(range(X.shape[1]))
tree = tree_generate(np.column_stack((X, y)), A)

# 预测
predictions = predict(tree, X)
print("Predictions:", predictions)
print("Actual:", y)