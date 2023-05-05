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

def train(X_train, y_train, X_val, y_val):
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

            # Mini-batch gradient descent
            grad = X_batch.T.dot(y_hat_batch - y_batch) / len(y_batch)
            w -= alpha * grad + decay * w

        # monitor model behavior after each epoch
        # 1. Compute the training loss by averaging loss_this_epoch
        loss_train = loss_this_epoch / N_train
        losses_train.append(loss_train)
        # 2. Perform validation on the validation set by the risk
        _, _, risk_val = predict(X_val, w, y_val)
        risks_val.append(risk_val/N_val)
        # 3. Keep track of the best validation epoch, risk, and the weights
        if risk_val < risk_best:
            w_best = w.copy()
            risk_best = risk_val
            epoch_best = epoch

    # Return some variables as needed
    return w_best, risk_best, epoch_best, losses_train, risks_val

# Load data
with open("housing.pkl", "rb") as f:
    (X, y) = pickle.load(f)

# X: sample x dimension
# y: sample x 1

X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

# Augment feature
X_ = np.concatenate((np.ones([X.shape[0], 1]), X), axis=1)
# X_: Nsample x (d+1)


# normalize features:
mean_y = np.mean(y)
std_y = np.std(y)

y = (y - np.mean(y)) / np.std(y)

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

# setting
alpha = 0.001       # learning rate
batch_size = 10     # batch size
MaxIter = 100        # Maximum iteration
decay = 0.0          # weight decay


# Train the model
w_best, risk_best, epoch_best, losses_train, risks_val = train(X_train, y_train, X_val, y_val)

# Report numbers
print("Best validation epoch: {}".format(epoch_best))
print("Validation performance (risk) in best epoch: {:.4f}".format(risk_best))
_, _, risk_test = predict(X_test, w_best, y_test)
print("Test performance (risk) in best epoch: {:.4f}".format(risk_test))

# Draw plots
plt.plot(range(MaxIter), losses_train)
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.show()

plt.plot(range(MaxIter), risks_val)
plt.xlabel('Epoch')
plt.ylabel('Validation Risk')
plt.show()

# plt.figure(figsize=(10, 5))
# plt.plot(losses_train, label="Training Loss")
# plt.plot(risks_val, label="Validation Risk")
# plt.xlabel("Epoch")
# plt.ylabel("Loss/Risk")
# plt.legend()
# plt.title("Learning Curve")
# plt.show()