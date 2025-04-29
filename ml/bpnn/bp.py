import numpy as np
import os
import pickle
import tkinter as tk
from PIL import Image, ImageDraw
from sklearn.preprocessing import OneHotEncoder


class BP:
    def __init__(self, layers, lr=0.001, epoches=100):
        self.layers = layers
        self.lr = lr
        self.epoches = epoches

        # 初始化参数
        self.weights = []
        self.biases = []
        for i in range(len(layers) - 1):
            limit = np.sqrt(2 / layers[i])  # He初始化
            self.weights.append(np.random.randn(layers[i], layers[i + 1]) * limit)
            self.biases.append(np.zeros(layers[i + 1]))

        # Adam优化器参数
        self.m_w = [np.zeros_like(w) for w in self.weights]
        self.v_w = [np.zeros_like(w) for w in self.weights]
        self.m_b = [np.zeros_like(b) for b in self.biases]
        self.v_b = [np.zeros_like(b) for b in self.biases]
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.t = 0

    @staticmethod
    def relu(x):
        return 1/(1+np.exp(-x))

    @staticmethod
    def relu_derivative(x):
        return x * (1 - x)

    @staticmethod
    def softmax(x):
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)

    def forward(self, X):
        # 确保输入是2D数组
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        self.layer_outputs = [X]
        for i in range(len(self.weights) - 1):
            z = np.dot(self.layer_outputs[-1], self.weights[i]) + self.biases[i]
            self.layer_outputs.append(self.relu(z))

        # 输出层使用softmax
        z = np.dot(self.layer_outputs[-1], self.weights[-1]) + self.biases[-1]
        self.layer_outputs.append(self.softmax(z))
        return self.layer_outputs[-1]

    def backward(self, y_true):
        deltas = []
        error = self.layer_outputs[-1] - y_true
        deltas.append(error)

        for i in range(len(self.weights) - 1, 0, -1):
            error = np.dot(deltas[0], self.weights[i].T)
            delta = error * self.relu_derivative(self.layer_outputs[i])
            deltas.insert(0, delta)

        return deltas

    def update_weights(self, deltas):
        self.t += 1
        for i in range(len(self.weights)):
            # 计算梯度
            grad_w = np.dot(self.layer_outputs[i].T, deltas[i]) / deltas[i].shape[0]
            grad_b = np.mean(deltas[i], axis=0)

            # Adam更新
            self.m_w[i] = self.beta1 * self.m_w[i] + (1 - self.beta1) * grad_w
            self.v_w[i] = self.beta2 * self.v_w[i] + (1 - self.beta2) * (grad_w ** 2)
            m_w_hat = self.m_w[i] / (1 - self.beta1 ** self.t)
            v_w_hat = self.v_w[i] / (1 - self.beta2 ** self.t)
            self.weights[i] -= self.lr * m_w_hat / (np.sqrt(v_w_hat) + self.epsilon)

            self.m_b[i] = self.beta1 * self.m_b[i] + (1 - self.beta1) * grad_b
            self.v_b[i] = self.beta2 * self.v_b[i] + (1 - self.beta2) * (grad_b ** 2)
            m_b_hat = self.m_b[i] / (1 - self.beta1 ** self.t)
            v_b_hat = self.v_b[i] / (1 - self.beta2 ** self.t)
            self.biases[i] -= self.lr * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)

    def train(self, X_train, y_train, batch_size=64):
        for epoch in range(self.epoches):
            # 随机分批
            indices = np.random.permutation(len(X_train))
            for start in range(0, len(X_train), batch_size):
                batch_idx = indices[start:start + batch_size]
                X_batch = X_train[batch_idx]
                y_batch = y_train[batch_idx]

                # 前向传播
                pred = self.forward(X_batch)

                # 反向传播
                deltas = self.backward(y_batch)
                self.update_weights(deltas)

            # 验证
            if epoch % 5 == 0:
                pred = self.forward(X_train[:1000])
                loss = -np.mean(y_train[:1000] * np.log(pred + 1e-10))
                acc = np.mean(np.argmax(pred, axis=1) == np.argmax(y_train[:1000], axis=1))
                print(f"Epoch {epoch}: Loss={loss:.4f}, Acc={acc:.4f}")

    def predict(self, X):
        # 确保输入始终是2D数组
        X = np.atleast_2d(X)
        return np.argmax(self.forward(X), axis=1)[0]  # 取第一个结果

    def save_model(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load_model(path):
        with open(path, 'rb') as f:
            return pickle.load(f)


class DigitRecognizerApp:
    def __init__(self, master, model):
        self.master = master
        self.model = model
        master.title("手写数字识别")

        # 画布设置
        self.canvas_width = 280
        self.canvas_height = 280
        self.brush_size = 15

        # 创建画布
        self.canvas = tk.Canvas(master, width=self.canvas_width, height=self.canvas_height, bg="black")
        self.canvas.pack()

        # 绑定鼠标事件
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.predict)

        # 预测结果显示
        self.prediction_label = tk.Label(master, text="请在上面画板书写数字", font=("Arial", 16))
        self.prediction_label.pack()

        # 控制按钮
        self.button_frame = tk.Frame(master)
        self.button_frame.pack()

        self.clear_button = tk.Button(self.button_frame, text="清除", command=self.clear_canvas)
        self.clear_button.pack(side=tk.LEFT, padx=10)

        self.exit_button = tk.Button(self.button_frame, text="退出", command=master.quit)
        self.exit_button.pack(side=tk.RIGHT, padx=10)

        # 绘图相关
        self.image = Image.new("L", (self.canvas_width, self.canvas_height), 0)
        self.draw = ImageDraw.Draw(self.image)

    def paint(self, event):
        x, y = event.x, event.y
        self.canvas.create_oval(x - self.brush_size, y - self.brush_size,
                                x + self.brush_size, y + self.brush_size,
                                fill="white", outline="white")
        self.draw.ellipse([x - self.brush_size, y - self.brush_size,
                           x + self.brush_size, y + self.brush_size],
                          fill=255)

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (self.canvas_width, self.canvas_height), 0)
        self.draw = ImageDraw.Draw(self.image)
        self.prediction_label.config(text="画板已清除，请重新书写")

    def predict(self, event):
        # 将画布图像转换为MNIST格式
        img = self.image.resize((28, 28), Image.LANCZOS)
        img_array = np.array(img).reshape(1, 784).astype(np.float32) / 255.0

        # 预测
        prediction = self.model.predict(img_array)
        proba = self.model.forward(img_array)
        confidence = np.max(proba) * 100

        self.prediction_label.config(
            text=f"预测数字: {prediction} (置信度: {confidence:.1f}%)"
        )


def load_mnist(path, kind='train'):
    """从本地加载MNIST数据"""
    labels_path = os.path.join(path, f'{kind}-labels-idx1-ubyte')
    images_path = os.path.join(path, f'{kind}-images-idx3-ubyte')

    with open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)

    with open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), 784)

    return images, labels


def preprocess_data():
    # 从本地加载数据
    data_dir = 'data/MNIST/raw'
    X_train, y_train = load_mnist(data_dir, kind='train')
    X_test, y_test = load_mnist(data_dir, kind='t10k')

    # 数据预处理
    X_train = X_train.astype(np.float32) / 255.0
    X_test = X_test.astype(np.float32) / 255.0

    # One-hot编码
    encoder = OneHotEncoder(sparse_output=False)
    y_train = encoder.fit_transform(y_train.reshape(-1, 1))
    y_test = encoder.transform(y_test.reshape(-1, 1))

    return X_train, y_train, X_test, y_test, encoder


if __name__ == "__main__":
    # 数据预处理
    X_train, y_train, X_test, y_test, encoder = preprocess_data()

    # 创建并训练网络
    model_path = 'mnist_bp_model.pkl'

    if not os.path.exists(model_path):
        print("Training new model...")
        nn = BP([784, 256, 128,72,36, 10], lr=0.01, epoches=10)
        nn.train(X_train[:20000], y_train[:20000])
        nn.save_model(model_path)
        print(f"Model saved to {model_path}")
    else:
        print("Loading existing model...")
        nn = BP.load_model(model_path)

        op = input("是否继续训练(y/n)：")
        if op.lower() == "y":
            num = int(input("继续训练多少轮: "))
            nn.epoches = num
            nn.train(X_train[:10000], y_train[:10000])
            nn.save_model(model_path)

    # 测试网络
    correct = 0
    test_samples = 1000  # 测试样本数量
    for x, y in zip(X_test[:test_samples], y_test[:test_samples]):
        pred = nn.predict(x)
        true_label = np.argmax(y)
        if pred == true_label:
            correct += 1

    print(f"\nTest Accuracy: {correct / test_samples * 100:.2f}%")

    # 启动交互式画板
    root = tk.Tk()
    app = DigitRecognizerApp(root, nn)
    root.mainloop()