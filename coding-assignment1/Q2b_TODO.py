# TODO: Copy from Q2a as needed

import numpy as np
import pickle
import matplotlib.pyplot as plt


def predict(X, w, y=None):
    y_hat = X.dot(w)
    if y is not None:
        loss = np.mean((y_hat - y) ** 2)/2
    else:
        loss = None
    if y is not None:
        y_denorm = y*np.std(y) + np.mean(y)
        y_hat_denorm = y_hat*np.std(y_hat)+np.mean(y_hat)
        risk = np.mean(abs(y_hat_denorm - y_denorm))
    else:
        risk = None
    return y_hat, loss, risk


def train(X_train, y_train, X_val, y_val, alpha, batch_size, MaxIter, decay):
    N_train = X_train.shape[0]
    N_val = X_val.shape[0]

    # initialization
    w = np.zeros([X_train.shape[1], 1])
    # w: (d+1)x1

    losses_train = []
    risks_val = []

    w_best = None
    risk_best = 10000
    epoch_best = 0

    for epoch in range(MaxIter):

        loss_this_epoch = 0
        for b in range(int(np.ceil(N_train/batch_size))):

            X_batch = X_train[b*batch_size: (b+1)*batch_size]
            y_batch = y_train[b*batch_size: (b+1)*batch_size]

            y_hat_batch, loss_batch, _ = predict(X_batch, w, y_batch)
            loss_this_epoch += loss_batch

            # Mini-batch gradient descent with L2 regularization
            grad = X_batch.T.dot(y_hat_batch - y_batch) / len(y_batch) + decay * w
            grad[-1] = grad[-1] - decay * w[-1]
            w -= alpha * grad

        # monitor model behavior after each epoch
        # 1. Compute the training loss by averaging loss_this_epoch
        loss_train = loss_this_epoch / N_train
        losses_train.append(loss_train)
        # 2. Perform validation on the validation set by the risk
        _, _, risk_val = predict(X_val, w, y_val)
        risks_val.append(risk_val)
        # 3. Keep track of the best validation epoch, risk, and the weights
        if risk_val < risk_best:
            w_best = w.copy()
            risk_best = risk_val
            epoch_best = epoch

    # Return some variables as needed
    return w_best, risk_best, epoch_best, losses_train, risks_val


# Main code starts here
# Load data
with open("housing.pkl", "rb") as f:
    (X, y) = pickle.load(f)

# X: sample x dimension
# y: sample x 1

X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
# Augment feature
X_ = np.concatenate((np.ones([X.shape[0], 1]), X), axis=1)
# X_: Nsample x (d+1)
X_ = np.concatenate((X_, X**2), axis=1)

# Normalize y
mean_y = np.mean(y)
std_y = np.std(y)
y = (y - mean_y) / std_y

# Randomly shuffle the data
np.random.seed(314)
np.random.shuffle(X_)
np.random.seed(314)
np.random.shuffle(y)

X_train = X_[:300]
y_train = y[:300]

X_val = X_[300:400]
y_val = y[300:400]

X_test = X_[400:]
y_test = y[400:]

#####################
# Setting
alpha = 0.001      # Learning rate
batch_size = 10    # Batch size
MaxIter = 100      # Maximum iteration
decay = 0.01       # L2 regularization parameter
lamda_values = [3, 1, 0.3, 0.1, 0.03, 0.01]

risks_val = []
for lamda in lamda_values:
    w_best, risk_best, epoch_best, losses_train, risks_val_lamda = train(X_train, y_train, X_val, y_val, alpha, batch_size, MaxIter, lamda)
    risks_val.append(risk_best)

# Report the best hyperparameter
best_idx = np.argmin(risks_val)
best_lamda = lamda_values[best_idx]
print("Best lambda:", best_lamda)

# # Train again with the best hyperparameter and compute the required outputs
# w_best, risk_best, epoch_best, losses_train, risks_val_best = train(X_train, y_train, X_val, y_val, alpha, batch_size, MaxIter, best_lamda)

# Compute the required outputs on the test set
# Plot learning curves
# Report numbers
print("Best validation epoch: {}".format(epoch_best))
print("Validation performance (risk) in best epoch: {}".format(risk_best))
_, _, risk_test = predict(X_test, w_best, y_test)
print("Test performance (risk) in best epoch: {}".format(risk_test))

# Draw plots
plt.plot(range(MaxIter), losses_train)
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.show()

plt.plot(range(MaxIter), risks_val_lamda)
plt.xlabel('Epoch')
plt.ylabel('Validation Risk')
plt.show()

 
