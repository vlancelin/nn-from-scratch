import numpy as np
from sklearn.datasets import fetch_openml


def load_data():
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X, y = mnist.data, mnist.target.astype(int)

    # Splitting in train/test
    X_train, y_train = X[:60000], y[:60000]
    X_test, y_test = X[60000:], y[60000:]

    # Normalizing the features 
    X_train, X_test = X_train/255, X_test/255

    # One hot encoding the train target 
    onehot_y_train = np.zeros([60000,10])
    onehot_y_train[np.arange(60000),y_train] =  1

    return X_train, onehot_y_train, X_test, y_test


X_train, onehot_y_train, X_test, y_test = load_data()

