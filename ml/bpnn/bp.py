from imageio.v2 import sizes
import numpy as np

def sigmoid(x):
    return 1.0/(1+np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x)*(1-sigmoid(x))

class BP:
    def __init__(self, input_num, hidden_num, output_num):
        self.num_layers=len(sizes)
        self.sizes=sizes
        self.biases=[np.random.randn(n,1) for n in sizes[1:]]
        self.weights=[np.random.randn(r,c) for c ,r in zip(sizes[:-1],sizes[1:])]
    def feed_forward(self,a):
        for b,w in zip(self.biases,self.weights):
            a=sigmoid(np.dot(w,a)+b)
        return a
    # def MSGD(self,train_data,epochs,mini_batch_size,eta,test_data=None):


bpnn=BP(10,12,10)