
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

# Loading the digits dataset
digits = load_digits()

# Split the data into training, validation, and testing sets
x_train, x_val, y_train, y_val = train_test_split(digits.data, digits.target, test_size=0.4, random_state=42)
x_val, x_test, y_val, y_test = train_test_split(x_val, y_val, test_size=0.5, random_state=42)

# Scaling the data to [0, 1]
x_train = x_train / 16.0
x_val = x_val / 16.0
x_test = x_test / 16.0

# Training a logistic regression classifier for multiple epochs
logistic = LogisticRegression(max_iter=1, random_state=42)
n_epochs = 10
train_losses = []
val_accuracies = []
for epoch in range(n_epochs):
    logistic.fit(x_train, y_train)
    y_pred_train = logistic.predict(x_train)
    train_loss = np.mean(y_pred_train != y_train)
    train_losses.append(train_loss)
    y_pred_val = logistic.predict(x_val)
    val_accuracy = np.mean(y_pred_val == y_val)
    val_accuracies.append(val_accuracy)
    print(f'Epoch {epoch + 1}/{n_epochs}: Train Loss = {train_loss:.4f}, Val Accuracy = {val_accuracy:.4f}')
    logistic.max_iter += 1

# Plotting the training loss and validation accuracy according to the epoch graph
fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Train Loss', color=color)
ax1.plot(range(1, n_epochs + 1), train_losses, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()

color = 'tab:blue'
ax2.set_ylabel('Val Accuracy', color=color)
ax2.plot(range(1, n_epochs + 1), val_accuracies, color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
plt.show()