import numpy as np

class NeuralNetwork:
    def __init__(self, hidden_layer=10, seed = None):
        if seed is not None: 
            np.random.seed(seed)

        self.W1 = np.random.rand(hidden_layer, 784)
        self.b1 = np.random.rand(hidden_layer, 1)

        self.W2 = np.random.rand(10, hidden_layer)
        self.b2 = np.random.rand(10, 1)

    def relu(self,x):
        return np.maximum(0, x)

    def softmax(self,x):
        return np.exp(x)/np.sum(np.exp(x), axis = 0)

    def forward(self, X):

        self.z1 = np.matmul(self.W1,X.T) + self.b1
        self.a1 = self.relu(self.z1)

        self.z2 = np.matmul(self.W2,self.a1) + self.b2
        self.a2 = self.softmax(self.z2)

        return self.a2
    
    def get_accuracy(self, gt, pred):
        predictions = np.argmax(pred, axis=0)
        return round((np.mean(predictions == gt) * 100), 2)

        






