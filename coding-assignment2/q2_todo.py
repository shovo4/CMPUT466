# -*- coding: utf-8 -*-
#reference: onehot --> https://gist.github.com/awjuliani/5ce098b4b76244b7a9e3#file-softmax-ipynb
import numpy as np
import struct
import matplotlib.pyplot as plt
from scipy.special import expit
import scipy
import scipy.sparse


def readMNISTdata():
    with open('t10k-images.idx3-ubyte', 'rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        nrows, ncols = struct.unpack(">II", f.read(8))
        test_data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        test_data = test_data.reshape((size, nrows*ncols))

    with open('t10k-labels.idx1-ubyte', 'rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        test_labels = np.fromfile(
            f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        test_labels = test_labels.reshape((size, 1))

    with open('train-images.idx3-ubyte', 'rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        nrows, ncols = struct.unpack(">II", f.read(8))
        train_data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        train_data = train_data.reshape((size, nrows*ncols))

    with open('train-labels.idx1-ubyte', 'rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        train_labels = np.fromfile(
            f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        train_labels = train_labels.reshape((size, 1))

    # augmenting a constant feature of 1 (absorbing the bias term)
    train_data = np.concatenate(
        (np.ones([train_data.shape[0], 1]), train_data), axis=1)
    test_data = np.concatenate(
        (np.ones([test_data.shape[0], 1]),  test_data), axis=1)
    _random_indices = np.arange(len(train_data))
    np.random.shuffle(_random_indices)
    train_labels = train_labels[_random_indices]
    train_data = train_data[_random_indices]

    X_train = train_data[:50000] / 256
    t_train = train_labels[:50000]

    X_val = train_data[50000:] / 256
    t_val = train_labels[50000:]

    return X_train, t_train, X_val, t_val, test_data / 256, test_labels


def predict(X, W, t=None):
    # X_new: Nsample x (d+1)
    # W: (d+1) x K
    z = np.dot(X, W)
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    y = exp_z / np.sum(exp_z, axis=1, keepdims=True) #softmax
    t_hat = np.argmax(y, axis=1)
    len = t.shape[0]
    one_hot = scipy.sparse.csr_matrix((np.ones(len), (t.flatten(), np.array(range(len))))) 
    t_one_hot = np.array(one_hot.todense()).T #onehot
    loss = -1/len * np.sum(t_one_hot * np.log(y))
    acc = np.mean(t_hat == t.flatten())
    return y, t_hat, loss, acc

def train(X_train, y_train, X_val, t_val):
    N_train = X_train.shape[0]
    N_val = X_val.shape[0]

    # Initialize weight matrix
    d = X_train.shape[1]
    K = len(np.unique(y_train))
    W = np.random.randn(d, K) * 0.001

    # Training loop
    epoch_best = 0
    acc_best = 0
    W_best = W.copy()
    train_losses = []
    valid_accs = []

    # print("{: >10} {: >20} {: >10}".format("epoch", "loss", "accuracy"))
    for epoch in range(MaxEpoch):

        # SGD with mini-batches
        for b in range(int(np.ceil(N_train/batch_size)) ):
            X_batch = X_train[b*batch_size : (b+1)*batch_size]
            y_batch = y_train[b*batch_size : (b+1)*batch_size]

            # Compute gradient
            oneshot = scipy.sparse.csr_matrix((np.ones(len(y_batch)), (y_batch.flatten(), np.array(range(len(y_batch))))))
            y_onehot = np.array(oneshot.todense()).T
            y_hat, _, _, _ = predict(X_batch, W, y_batch)
            grad_W =  X_batch.T @ (y_hat - y_onehot)

            # Update weights
            W -= alpha * grad_W / batch_size
            W -= alpha * decay * W  # weight decay

        # Evaluate on training set
        y_pred, _, train_loss, train_acc = predict(X_train, W, y_train)
        train_losses.append(train_loss)
        
        # Evaluate on validation set
        _, _, _, valid_acc = predict(X_val, W, t_val)
        valid_accs.append(valid_acc)

        # Check if best epoch so far
        # print("{: >10} {: >20} {: >10}".format(epoch, train_loss, valid_acc))
        if valid_acc >= acc_best:
            epoch_best = epoch
            acc_best = valid_acc
            W_best = W.copy()

    return epoch_best, acc_best, W_best, train_losses, valid_accs




##############################
# Main code starts here
X_train, t_train, X_val, t_val, X_test, t_test = readMNISTdata()


print(X_train.shape, t_train.shape, X_val.shape,
      t_val.shape, X_test.shape, t_test.shape)


N_class = 10

alpha = 0.1      # learning rate
batch_size = 100    # batch size
MaxEpoch = 50        # Maximum epoch
decay = 0.          # weight decay


# TODO: report 3 number, plot 2 curves
epoch_best, acc_best,  W_best, train_losses, valid_accs = train(X_train, t_train, X_val, t_val)

_, _, _, acc_test = predict(X_test, W_best, t_test)
# Report numbers and draw plots as required.
print("Epoch for best validation performance = ", epoch_best)
print("Validation performance = ",acc_best)
print("Test performance = ", acc_test)

# used plt code from https://www.geeksforgeeks.org/how-to-plot-a-smooth-curve-in-matplotlib/ 
# to plot the curve

x_axis = range(MaxEpoch)

plt.plot(train_losses)
plt.title("Training loss")
plt.xlabel("Epoch")
plt.ylabel("Cross-entropy loss")
plt.show()

plt.plot(valid_accs)
plt.title("Validation accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.show()






