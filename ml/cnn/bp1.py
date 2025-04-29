import numpy as np


class BP:
    def __init__(self, layers, lr, epoches):
        self.learn_rate = lr
        self.epoches = epoches
        self.layers = layers
        self.weights = []
        self.biases = []  # 修正变量名从bars改为biases

        # 初始化权重
        for i in range(len(layers) - 1):
            # 使用numpy的随机函数更高效地初始化权重
            weight = np.random.rand(layers[i], layers[i + 1]) * 2 - 1  # 范围在-1到1之间
            self.weights.append(weight)

        # 初始化偏置
        for i in range(1, len(layers)):
            bias = np.random.rand(layers[i]) * 2 - 1
            self.biases.append(bias)

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_derivative(x):
        return x * (1 - x)

    def forward(self, inputs):
        inputs = np.array(inputs)
        self.layer_outputs = [inputs]  # 保存各层输出，用于反向传播

        for i in range(len(self.weights)):
            # 计算加权和
            weighted_sum = np.dot(self.layer_outputs[-1], self.weights[i]) + self.biases[i]
            # 应用激活函数
            output = self.sigmoid(weighted_sum)
            self.layer_outputs.append(output)

        return self.layer_outputs[-1]

    def backward(self, expected):
        expected = np.array(expected)
        errors = []
        deltas = []

        # 计算输出层误差
        error = expected - self.layer_outputs[-1]
        errors.insert(0, error)

        # 计算输出层delta
        delta = error * self.sigmoid_derivative(self.layer_outputs[-1])
        deltas.insert(0, delta)

        # 反向传播隐藏层误差
        for i in range(len(self.weights) - 1, 0, -1):
            error = deltas[0].dot(self.weights[i].T)
            errors.insert(0, error)
            delta = error * self.sigmoid_derivative(self.layer_outputs[i])
            deltas.insert(0, delta)

        return deltas

    def update_weights(self, deltas):
        for i in range(len(self.weights)):
            # 更新权重
            self.weights[i] += self.learn_rate * np.dot(self.layer_outputs[i].reshape(-1, 1),
                                                        deltas[i].reshape(1, -1))
            # 更新偏置
            self.biases[i] += self.learn_rate * deltas[i]

    def train(self, training_inputs, training_outputs):
        for epoch in range(self.epoches):
            total_error = 0
            for inputs, outputs in zip(training_inputs, training_outputs):
                # 前向传播
                self.forward(inputs)

                # 反向传播
                deltas = self.backward(outputs)

                # 更新权重
                self.update_weights(deltas)

                # 计算误差
                total_error += np.mean(np.abs(outputs - self.layer_outputs[-1]))

            if epoch % 1000 == 0:
                print(f"Epoch {epoch}, Error: {total_error / len(training_inputs)}")

    def predict(self, inputs):
        return self.forward(inputs)

def load_data(filename):
    data = []
    with open(filename,'r') as f:

        for row in f:
            row=row.replace('[','').replace(']','').split()
            data.append([float(x) for x in row])
    return data
# 测试代码
if __name__ == "__main__":



    data = load_data("train.txt")
    for i in range(0,len(data)):
        if data[i][2]==-1:
            data[i][2]=0
    data=np.array(data)
    X=data[:,:2]
    print(X)
    y=data[:,-1]
    print(y)
    # for i in range(0,len(y)):



    # 创建网络 (2输入, 4隐藏神经元, 1输出)
    nn = BP([2, 4,8,16, 1], lr=0.1, epoches=100000)

    # 训练网络
    nn.train(X, y)



    # 测试网络
    print("\nTest Results:")
    data_test=load_data('test.txt')

    data_test = np.array(data_test)
    X_test = data[:, :2]
    print(X_test)
    y_test = data[:, -1]
    print(y_test)
    for x,y in X_test,y_test:
        prediction = nn.predict(x)
        print(f"Input: {x}, Output: {prediction}, Rounded: {np.round(prediction)},label:{y}")