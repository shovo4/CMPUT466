from utils import plot_data, generate_data
import numpy as np


"""
Documentation:

Function generate() takes as input "A" or "B", it returns X, t.
X is two dimensional vectors, t is the list of labels (0 or 1).    

Function plot_data(X, t, w=None, bias=None, is_logistic=False, figure_name=None)
takes as input paris of (X, t) , parameter w, and bias. 
If you are plotting the decision boundary for a logistic classifier, set "is_logistic" as True
"figure_name" specifies the name of the saved diagram.
"""


def train_logistic_regression(X, t):
    """
    Given data, train your logistic classifier.
    Return weight and bias
    """
    # initialize weights and bias to zeros
    n_samples, n_features = X.shape
    w = np.zeros((n_features))
    b = 0

    # hyperparameters
    alpha = 0.1
    epochs = 1000

    # gradient descent
    for epoch in range(epochs):
        # forward pass
        z = np.dot(X, w) + b
        y = 1 / (1 + np.exp(-z))

        # backward pass
        dz = y - t
        dw = 1/n_samples * np.dot(X.T, dz)
        db = 1/n_samples * np.sum(dz)

        # update parameters
        w = w - alpha * dw
        b = b - alpha * db
    return w, b


def predict_logistic_regression(X, w, b):
    """
    Generate predictions by your logistic classifier.
    """
    z = np.dot(X, w) + b
    y = 1 / (1 + np.exp(-z))
    t = np.round(y)
    return t


def train_linear_regression(X, t):
    """
    Given data, train your linear regression classifier.
    Return weight and bias
    """
    # add bias term to X
    X = np.hstack((X, np.ones((X.shape[0], 1))))

    # calculate weights using closed-form solution
    w = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(t)

    # separate weights and bias
    b = w[-1]
    w = w[:-1]
    return w, b


def predict_linear_regression(X, w, b):
    """
    Generate predictions by your logistic classifier.
    """
    z = np.dot(X, w) + b
    t = np.round(z)
    return t


def get_accuracy(t, t_hat):
    """
    Calculate accuracy,
    """
    acc = np.mean(t == t_hat)
    return acc


def main():
    # Dataset A
    # Linear regression classifier
    X, t = generate_data("A")
    w, b = train_linear_regression(X, t)
    t_hat = predict_linear_regression(X, w, b)
    print("Accuracy of linear regression on dataset A:", get_accuracy(t_hat, t))
    plot_data(X, t, w, b, is_logistic=False,
              figure_name='dataset_A_linear.png')

    # logistic regression classifier
    X, t = generate_data("A")
    w, b = train_logistic_regression(X, t)
    t_hat = predict_logistic_regression(X, w, b)
    print("Accuracy of logistic regression on dataset A:", get_accuracy(t_hat, t))
    plot_data(X, t, w, b, is_logistic=True,
              figure_name='dataset_A_logistic.png')

    # Dataset B
    # Linear regression classifier
    X, t = generate_data("B")
    w, b = train_linear_regression(X, t)
    t_hat = predict_linear_regression(X, w, b)
    print("Accuracy of linear regression on dataset B:", get_accuracy(t_hat, t))
    plot_data(X, t, w, b, is_logistic=False,
              figure_name='dataset_B_linear.png')

    # logistic regression classifier
    X, t = generate_data("B")
    w, b = train_logistic_regression(X, t)
    t_hat = predict_logistic_regression(X, w, b)
    print("Accuracy of logistic regression on dataset B:", get_accuracy(t_hat, t))
    plot_data(X, t, w, b, is_logistic=True,
              figure_name='dataset_B_logistic.png')


main()
