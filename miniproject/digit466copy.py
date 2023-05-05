# -*- coding: utf-8 -*-
"""digit466.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1W_F-DmNAIa-VtQ0ixukBR28KnXSnp6Wy
"""

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, log_loss
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

# Loading the digits dataset
digits = load_digits()

# Spliting the data into training, validation, and testing sets 40% for testing and 60% for training and validation
x_train, x_val, y_train, y_val = train_test_split(digits.data, digits.target, test_size=0.4, random_state=42)
x_val, x_test, y_val, y_test = train_test_split(x_val, y_val, test_size=0.5, random_state=42)

# Scaling the data to [0, 1]
x_train = x_train / 16.0
x_val = x_val / 16.0
x_test = x_test / 16.0

# Training a logistic regression classifier
logistic = LogisticRegression(max_iter=1000, random_state=42)
logistic.fit(x_train, y_train)

# Calculating training loss
loss_train_logistic = log_loss(y_train, logistic.predict_proba(x_train))
print('Logistic Regression Training Loss:', loss_train_logistic)

# Calculating validation accuracy
validation_accuracy = logistic.score(x_val, y_val)
print('Logistic Regression Validation Accuracy:', validation_accuracy)

# Predicting labels for test set using logistic regression
y_pred_logistic = logistic.predict(x_test)
accuracy_logistic = accuracy_score(y_test, y_pred_logistic)
print('Logistic Regression Test Accuracy:', accuracy_logistic)

# Hyperparameter tuning for Logistic Regression
hyperlogistic = LogisticRegression(max_iter=1000, random_state=42)
param_grid_logistic = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}
grid_search_logistic = GridSearchCV(hyperlogistic, param_grid_logistic, cv=5, scoring='accuracy')
grid_search_logistic.fit(x_train, y_train)

best_logistic = grid_search_logistic.best_estimator_
print("Best logistic regression hyperparameters:", grid_search_logistic.best_params_)

# Predicting labels for test set using logistic regression with hyperparameters
y_pred_logistic_hyper = best_logistic.predict(x_test)
accuracy_logistic_hyper = accuracy_score(y_test, y_pred_logistic_hyper)
print('Logistic Regression with hyperparameters Test Accuracy:', accuracy_logistic_hyper)


# # Printing classification report for logistic regression
# print("Logistic Regression Classification Report:")
# print(classification_report(y_test, y_pred_logistic))
# # Print confusion matrix for logistic regression
# print("Logistic Regression Confusion Matrix:")
# print(confusion_matrix(y_test, y_pred_logistic))


# Training a linear regression model
linear = LinearRegression()
linear.fit(x_train, y_train)

# Predicting labels for test set using linear regression
y_pred_linear = np.round(linear.predict(x_test))
accuracy_linear = accuracy_score(y_test, y_pred_linear)
print('Linear Regression Accuracy:', accuracy_linear)



# Training a support vector machine classifier
svm = SVC(kernel='rbf', gamma='scale', random_state=42)
svm.fit(x_train, y_train)

# Predict labels for test set using support vector machine
y_pred_svm = svm.predict(x_test)
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print('SVM Accuracy:', accuracy_svm)


# Hyperparameter tuning for SVC
hypersvm = SVC(random_state=42)
param_grid_svm = {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto']}
grid_search_svm = GridSearchCV(hypersvm, param_grid_svm, cv=5, scoring='accuracy')
grid_search_svm.fit(x_train, y_train)

best_svm = grid_search_svm.best_estimator_
print("Best SVC hyperparameters:", grid_search_svm.best_params_)

# Predict labels for test set using support vector machine
y_pred_svm_hyper = best_svm.predict(x_test)
accuracy_svm_hyper = accuracy_score(y_test, y_pred_svm_hyper)
print('SVM with hyperparameters Test Accuracy:', accuracy_svm_hyper)

# # Print classification report for SVM
# print("SVM Classification Report:")
# print(classification_report(y_test, y_pred_svm))
# # Print confusion matrix for SVM
# print("SVM Confusion Matrix:")
# print(confusion_matrix(y_test, y_pred_svm))


# Train a decision tree classifier
tree = DecisionTreeClassifier(random_state=42)
tree.fit(x_train, y_train)

# Predict labels for test set using decision tree
y_pred_tree = tree.predict(x_test)
accuracy_tree = accuracy_score(y_test, y_pred_tree)
print('Decision Tree Accuracy:', accuracy_tree)

# # Print classification report for decision tree
# print("Decision Tree Classification Report:")
# print(classification_report(y_test, y_pred_tree))
# # Print confusion matrix for decision tree
# print("Decision Tree Confusion Matrix:")
# print(confusion_matrix(y_test, y_pred_tree))

# Train a KNN classifier
knn = KNeighborsClassifier()
knn.fit(x_train, y_train)

# Predict labels for test set using KNN
y_pred_knn = knn.predict(x_test)
accuracy_knn = accuracy_score(y_test, y_pred_knn)
print('KNN Accuracy:', accuracy_knn)

# Hyperparameter tuning for KNN Classifier
knn_hyper = KNeighborsClassifier()
param_grid_knn = {'n_neighbors': [1, 3, 5, 7, 9]}
grid_search_knn = GridSearchCV(knn_hyper, param_grid_knn, cv=5, scoring='accuracy')
grid_search_knn.fit(x_train, y_train)

best_knn = grid_search_knn.best_estimator_
print("Best KNN hyperparameters:", grid_search_knn.best_params_)

# Predict labels for test set using KNN with hyperparameters
y_pred_knn_hyper = best_knn.predict(x_test)
accuracy_knn_hyper = accuracy_score(y_test, y_pred_knn_hyper)
print('KNN with hyperparameters Test Accuracy:', accuracy_knn_hyper)

# # Print classification report for KNN
# print("KNN Classification Report:")
# print(classification_report(y_test, y_pred_knn))
# # Print confusion matrix for KNN
# print("KNN Confusion Matrix:")
# print(confusion_matrix(y_test, y_pred_knn))

# Training a logistic regression classifier using stochastic gradient descent (SGD)
sgd = LogisticRegression(max_iter=1000, solver='saga', random_state=42)
sgd.fit(x_train, y_train)

# Predicting labels for test set using SGD logistic regression
y_pred_sgd = sgd.predict(x_test)
accuracy_sgd = accuracy_score(y_test, y_pred_sgd)
print('SGD Logistic Regression Accuracy:', accuracy_sgd)

# # Printing classification report for SGD logistic regression
# print("SGD Logistic Regression Classification Report:")
# print(classification_report(y_test, y_pred_sgd))

# # Print confusion matrix for SGD logistic regression
# print("SGD Logistic Regression Confusion Matrix:")
# print(confusion_matrix(y_test, y_pred_sgd))


# Train a neural network classifier
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, activation='relu', solver='adam', random_state=42)
mlp.fit(x_train, y_train)

# Predict labels for test set using neural network
y_pred_mlp = mlp.predict(x_test)
accuracy_mlp = accuracy_score(y_test, y_pred_mlp)
print('Neural Network Accuracy:', accuracy_mlp)

# # Print classification report for neural network

# print("Neural Network Classification Report:")
# print(classification_report(y_test, y_pred_mlp))

# # Print confusion matrix for neural network
# print("Neural Network Confusion Matrix:")
# print(confusion_matrix(y_test, y_pred_mlp))
# Evaluate models on validation set
# Evaluate models on test set
models = [accuracy_logistic, accuracy_linear, accuracy_svm, accuracy_tree, accuracy_knn, accuracy_sgd, accuracy_mlp]
model_names = ['Logistic Regression', 'Linear Regression', 'SVM' , 'KNN', 'SGD Logistic Regression', 'Neural Network']
test_accuracies = []

for model, name in zip(models, model_names):
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    test_accuracies.append(accuracy)
    print(f"{name} Test Accuracy: {accuracy}")
val_accuracies = []

for model, name in zip(models, model_names):
    y_pred_val = model.predict(x_val)
    accuracy_val = accuracy_score(y_val, y_pred_val)
    val_accuracies.append(accuracy_val)
    print(f"{name} Validation Accuracy: {accuracy_val}")

models = ['Logistic Regression', 'Linear Regression', 'SVM' , 'Decision Tree' , 'KNN', 'SGD Logistic Regression', 'Neural Network']
accuracies = [accuracy_logistic, accuracy_linear, accuracy_svm, accuracy_tree, accuracy_knn, accuracy_sgd, accuracy_mlp]

plt.barh(models, accuracies)
plt.title('Accuracy Scores for Different Models')
plt.xlabel('Accuracy')
plt.show()

models = ['Logistic Regression Hyperparameters', 'Linear Regression', 'SVM Hyperparameters' , 'Decision Tree' , 'KNN Hyperparameters', 'SGD Logistic Regression', 'Neural Network']
accuracies = [accuracy_logistic_hyper, accuracy_linear, accuracy_svm_hyper, accuracy_tree, accuracy_knn_hyper, accuracy_sgd, accuracy_mlp]

plt.barh(models, accuracies)
plt.title('Accuracy Scores for Different Models')
plt.xlabel('Accuracy')
plt.show()