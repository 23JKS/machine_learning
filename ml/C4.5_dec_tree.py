import numpy as np
from collections import Counter
from graphviz import Digraph
dic={0:"age of the patient",1:"spectacle",2:"astigmatic",3:"tear production rate"}

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


def split_information(X_column):
    unique_values, counts = np.unique(X_column, return_counts=True)
    probabilities = counts / len(X_column)
    return -sum(p * np.log2(p) for p in probabilities)


def information_gain_ratio(X_column, y):
    parent_entropy = entropy(y)
    n = len(y)
    unique_values = np.unique(X_column)
    child_entropy = 0

    for value in unique_values:
        subset_indices = X_column == value
        subset_y = y[subset_indices]
        if len(subset_y) == 0:
            continue
        child_entropy += (len(subset_y) / n) * entropy(subset_y)

    info_gain = parent_entropy - child_entropy
    split_info = split_information(X_column)
    if split_info == 0:
        return 0
    return info_gain / split_info


def find_best_split(X, y):
    best_gain_ratio = 0
    best_feature = None

    for feature_idx in range(X.shape[1]):
        X_column = X[:, feature_idx]
        gain_ratio = information_gain_ratio(X_column, y)
        if gain_ratio > best_gain_ratio:
            best_gain_ratio = gain_ratio
            best_feature = feature_idx

    return best_feature, None  # 离散值特征不需要 threshold


def tree_generate(D, A, max_depth=5, depth=0):
    y = D[:, -1]
    if len(set(y)) == 1:  # 如果所有样本属于同一类别
        return Node(value=y[0])

    if len(A) == 0 or all(np.all(D[:, a] == D[0, a]) for a in A):  # 如果特征集为空或所有样本特征值相同
        majority_class = Counter(y).most_common(1)[0][0]
        return Node(value=majority_class)

    best_feature, _ = find_best_split(D[:, A], y)
    if best_feature is None:  # 如果没有找到最佳特征
        majority_class = Counter(y).most_common(1)[0][0]
        return Node(value=majority_class)

    feature_idx_in_A = A[best_feature]
    node = Node(feature=feature_idx_in_A)

    unique_values = np.unique(D[:, feature_idx_in_A])
    for value in unique_values:
        D_v = D[D[:, feature_idx_in_A] == value]
        if len(D_v) == 0:  # 如果子集为空
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
        # 如果特征值不在分支中，返回出现次数最多的类别
        return Counter([branch.value for branch in tree.branches.values()]).most_common(1)[0][0]


def predict(tree, X):
    return [predict_sample(tree, x) for x in X]


def tree_to_dict(tree):
    if tree.is_leaf():
        return {"value": tree.value}
    else:
        return {
            "feature": tree.feature,
            "branches": {value: tree_to_dict(branch) for value, branch in tree.branches.items()}
        }


def plot_tree(tree, graph=None, parent_name=None, edge_label=None):
    if graph is None:
        graph = Digraph()

    if tree.is_leaf():
        node_label = f"Class: {tree.value}"
    else:
        node_label = dic[tree.feature]

    node_name = str(id(tree))  # 使用节点对象的唯一 ID 作为节点名称
    graph.node(node_name, label=node_label)

    if parent_name is not None:
        graph.edge(parent_name, node_name, label=str(edge_label))

    if not tree.is_leaf():
        for value, branch in tree.branches.items():
            plot_tree(branch, graph, node_name, value)

    return graph


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


for i in range(0,4):
    print(dic[i],"的信息增益率：",information_gain_ratio(X[:, i],y))
# 构建决策树
A = list(range(X.shape[1]))
tree = tree_generate(np.column_stack((X, y)), A)

# 输出决策树结构（字典）
tree_dict = tree_to_dict(tree)


# 绘制决策树图形
graph = plot_tree(tree)
graph.render("decision_tree", format="png", cleanup=True)
print("Decision Tree saved as 'decision_tree.png'")