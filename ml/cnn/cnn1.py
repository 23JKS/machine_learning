import numpy as np
import os
from sklearn.preprocessing import OneHotEncoder


# 1. 确保数据存在
def ensure_mnist_data():
    mnist_dir = './data/MNIST/raw'
    required_files = [
        'train-images-idx3-ubyte',
        'train-labels-idx1-ubyte',
        't10k-images-idx3-ubyte',
        't10k-labels-idx1-ubyte'
    ]

    if not all(os.path.exists(os.path.join(mnist_dir, f)) for f in required_files):
        raise FileNotFoundError(
            "MNIST数据文件缺失。请下载以下文件并放入./data/mnist/目录:\n" +
            "\n".join(required_files)
        )


ensure_mnist_data()


# 2. 加载数据
def load_mnist(path, kind='train'):
    """加载MNIST数据"""
    labels_path = os.path.join(path, f'{kind}-labels-idx1-ubyte')
    images_path = os.path.join(path, f'{kind}-images-idx3-ubyte')

    with open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)

    with open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), 784)

    return images, labels


# 加载数据
X_train, y_train = load_mnist('data/MNIST/raw', kind='train')
X_test, y_test = load_mnist('data/MNIST/raw', kind='t10k')

# 预处理
X_train = (X_train / 255.0).reshape(-1, 1, 28, 28)
X_test = (X_test / 255.0).reshape(-1, 1, 28, 28)

# One-hot编码
encoder = OneHotEncoder(sparse_output=False)
y_train_onehot = encoder.fit_transform(y_train.reshape(-1, 1))
y_test_onehot = encoder.transform(y_test.reshape(-1, 1))

print("数据加载完成！")
print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")


# 2. 实现CNN核心组件

class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # 初始化卷积核 (out_channels, in_channels, kernel_size, kernel_size)
        self.weights = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.1
        self.bias = np.zeros(out_channels)

    def forward(self, x):
        batch_size, in_channels, in_h, in_w = x.shape
        out_h = (in_h - self.kernel_size + 2 * self.padding) // self.stride + 1
        out_w = (in_w - self.kernel_size + 2 * self.padding) // self.stride + 1

        # 添加padding
        if self.padding > 0:
            x_padded = np.zeros((batch_size, in_channels,
                                 in_h + 2 * self.padding,
                                 in_w + 2 * self.padding))
            x_padded[:, :, self.padding:-self.padding, self.padding:-self.padding] = x
        else:
            x_padded = x

        output = np.zeros((batch_size, self.out_channels, out_h, out_w))

        for b in range(batch_size):
            for c_out in range(self.out_channels):
                for c_in in range(self.in_channels):
                    for h in range(out_h):
                        for w in range(out_w):
                            h_start = h * self.stride
                            w_start = w * self.stride
                            h_end = h_start + self.kernel_size
                            w_end = w_start + self.kernel_size

                            window = x_padded[b, c_in, h_start:h_end, w_start:w_end]
                            output[b, c_out, h, w] += np.sum(window * self.weights[c_out, c_in])

                output[b, c_out] += self.bias[c_out]

        self.input = x  # 保存输入用于反向传播
        return output

    def backward(self, grad_output, lr=0.01):
        batch_size, in_channels, in_h, in_w = self.input.shape
        _, out_channels, out_h, out_w = grad_output.shape

        grad_input = np.zeros_like(self.input)
        grad_weights = np.zeros_like(self.weights)
        grad_bias = np.zeros_like(self.bias)

        # 添加padding
        if self.padding > 0:
            x_padded = np.zeros((batch_size, in_channels,
                                 in_h + 2 * self.padding,
                                 in_w + 2 * self.padding))
            x_padded[:, :, self.padding:-self.padding, self.padding:-self.padding] = self.input
        else:
            x_padded = self.input

        # 计算梯度
        for b in range(batch_size):
            for c_out in range(out_channels):
                for c_in in range(in_channels):
                    for h in range(out_h):
                        for w in range(out_w):
                            h_start = h * self.stride
                            w_start = w * self.stride
                            h_end = h_start + self.kernel_size
                            w_end = w_start + self.kernel_size

                            # 确保不越界
                            if h_end > x_padded.shape[2] or w_end > x_padded.shape[3]:
                                continue

                            window = x_padded[b, c_in, h_start:h_end, w_start:w_end]

                            # 计算权重梯度
                            grad_weights[c_out, c_in] += window * grad_output[b, c_out, h, w]

                            # 计算输入梯度 - 确保形状匹配
                            grad_input_window = self.weights[c_out, c_in] * grad_output[b, c_out, h, w]
                            if grad_input_window.shape != (h_end - h_start, w_end - w_start):
                                grad_input_window = grad_input_window[:h_end - h_start, :w_end - w_start]

                            grad_input[b, c_in, h_start:h_end, w_start:w_end] += grad_input_window

                grad_bias[c_out] += np.sum(grad_output[b, c_out])

        # 更新参数
        self.weights -= lr * grad_weights / batch_size
        self.bias -= lr * grad_bias / batch_size

        # 去掉padding部分的梯度
        if self.padding > 0:
            grad_input = grad_input[:, :, self.padding:-self.padding, self.padding:-self.padding]

        return grad_input


class MaxPool2D:
    def __init__(self, kernel_size=2, stride=2):
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        batch_size, channels, in_h, in_w = x.shape
        out_h = (in_h - self.kernel_size) // self.stride + 1
        out_w = (in_w - self.kernel_size) // self.stride + 1

        output = np.zeros((batch_size, channels, out_h, out_w))
        self.max_indices = np.zeros((batch_size, channels, out_h, out_w, 2), dtype=int)

        for b in range(batch_size):
            for c in range(channels):
                for h in range(out_h):
                    for w in range(out_w):
                        h_start = h * self.stride
                        w_start = w * self.stride
                        h_end = h_start + self.kernel_size
                        w_end = w_start + self.kernel_size

                        window = x[b, c, h_start:h_end, w_start:w_end]
                        output[b, c, h, w] = np.max(window)

                        # 记录最大值位置
                        max_idx = np.unravel_index(np.argmax(window), window.shape)
                        self.max_indices[b, c, h, w] = [h_start + max_idx[0], w_start + max_idx[1]]

        self.input_shape = x.shape
        return output

    def backward(self, grad_output):
        batch_size, channels, out_h, out_w = grad_output.shape
        grad_input = np.zeros(self.input_shape)

        for b in range(batch_size):
            for c in range(channels):
                for h in range(out_h):
                    for w in range(out_w):
                        max_h, max_w = self.max_indices[b, c, h, w]
                        grad_input[b, c, max_h, max_w] = grad_output[b, c, h, w]

        return grad_input


class ReLU:
    def forward(self, x):
        self.input = x
        return np.maximum(0, x)

    def backward(self, grad_output):
        grad_input = grad_output.copy()
        grad_input[self.input <= 0] = 0
        return grad_input


class Flatten:
    def forward(self, x):
        self.input_shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, grad_output):
        return grad_output.reshape(self.input_shape)


class Linear:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size) * 0.1
        self.bias = np.zeros(output_size)

    def forward(self, x):
        self.input = x
        return np.dot(x, self.weights) + self.bias

    def backward(self, grad_output, lr=0.01):
        grad_input = np.dot(grad_output, self.weights.T)
        grad_weights = np.dot(self.input.T, grad_output)
        grad_bias = np.sum(grad_output, axis=0)

        # 更新参数
        self.weights -= lr * grad_weights / grad_output.shape[0]
        self.bias -= lr * grad_bias / grad_output.shape[0]

        return grad_input


class SoftmaxCrossEntropy:
    def forward(self, x, y):
        # 计算softmax
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        self.softmax = exp_x / np.sum(exp_x, axis=1, keepdims=True)

        # 计算交叉熵损失
        self.y = y
        loss = -np.sum(y * np.log(self.softmax + 1e-10)) / x.shape[0]
        return loss

    def backward(self):
        return (self.softmax - self.y) / self.y.shape[0]


# 3. 构建CNN模型
class SimpleCNN:
    def __init__(self):
        self.conv1 = Conv2D(1, 32, kernel_size=3, padding=1)
        self.relu1 = ReLU()
        self.pool1 = MaxPool2D(kernel_size=2, stride=2)

        self.conv2 = Conv2D(32, 64, kernel_size=3, padding=1)
        self.relu2 = ReLU()
        self.pool2 = MaxPool2D(kernel_size=2, stride=2)

        self.flatten = Flatten()
        self.fc1 = Linear(64 * 7 * 7, 128)
        self.relu3 = ReLU()
        self.fc2 = Linear(128, 10)

        self.loss = SoftmaxCrossEntropy()

    def forward(self, x):
        x = self.conv1.forward(x)
        x = self.relu1.forward(x)
        x = self.pool1.forward(x)

        x = self.conv2.forward(x)
        x = self.relu2.forward(x)
        x = self.pool2.forward(x)

        x = self.flatten.forward(x)
        x = self.fc1.forward(x)
        x = self.relu3.forward(x)
        x = self.fc2.forward(x)
        return x

    def backward(self, lr=0.01):
        grad = self.loss.backward()

        grad = self.fc2.backward(grad, lr)
        grad = self.relu3.backward(grad)
        grad = self.fc1.backward(grad, lr)
        grad = self.flatten.backward(grad)

        grad = self.pool2.backward(grad)
        grad = self.relu2.backward(grad)
        grad = self.conv2.backward(grad, lr)

        grad = self.pool1.backward(grad)
        grad = self.relu1.backward(grad)
        grad = self.conv1.backward(grad, lr)


# 4. 训练模型
model = SimpleCNN()
batch_size = 64
num_epochs = 5
lr = 0.01

for epoch in range(num_epochs):
    total_loss = 0
    correct = 0
    total = 0

    # 随机打乱数据
    indices = np.random.permutation(len(X_train))
    X_train_shuffled = X_train[indices]
    y_train_shuffled = y_train_onehot[indices]

    for i in range(0, len(X_train), batch_size):
        # 获取小批量数据
        X_batch = X_train_shuffled[i:i + batch_size]
        y_batch = y_train_shuffled[i:i + batch_size]

        # 前向传播
        outputs = model.forward(X_batch)
        loss = model.loss.forward(outputs, y_batch)
        total_loss += loss

        # 计算准确率
        preds = np.argmax(outputs, axis=1)
        labels = np.argmax(y_batch, axis=1)
        correct += np.sum(preds == labels)
        total += len(labels)

        # 反向传播和参数更新
        model.backward(lr)

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / (len(X_train) / batch_size):.4f}, "
          f"Train Acc: {100 * correct / total:.2f}%")

# 5. 测试模型
correct = 0
total = 0

for i in range(0, len(X_test), batch_size):
    X_batch = X_test[i:i + batch_size]
    y_batch = y_test_onehot[i:i + batch_size]

    outputs = model.forward(X_batch)
    preds = np.argmax(outputs, axis=1)
    labels = np.argmax(y_batch, axis=1)

    correct += np.sum(preds == labels)
    total += len(labels)

print(f"Test Accuracy: {100 * correct / total:.2f}%")


# 6. 可视化测试结果
def plot_images(images, labels, preds, n=6):
    plt.figure(figsize=(12, 4))
    for i in range(n):
        plt.subplot(1, n, i + 1)
        plt.imshow(images[i][0], cmap='gray')
        plt.title(f"Label: {labels[i]}\nPred: {preds[i]}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()


# 获取一批测试数据
sample_indices = np.random.choice(len(X_test), 6)
X_sample = X_test[sample_indices]
y_sample = y_test[sample_indices]

# 预测
outputs = model.forward(X_sample)
preds = np.argmax(outputs, axis=1)

# 可视化
plot_images(X_sample, y_sample, preds)