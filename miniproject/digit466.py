import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, log_loss
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# Load the digits dataset
digits = load_digits()

# Split the data into training, validation, and testing sets
# 60% training, 20% validation, 20% testing
x_train, x_val, y_train, y_val = train_test_split(digits.data, digits.target, test_size=0.4, random_state=42) 
x_val, x_test, y_val, y_test = train_test_split(x_val, y_val, test_size=0.5, random_state=42)

# Scale the data to [0, 1]
x_train = x_train / 16.0
x_val = x_val / 16.0
x_test = x_test / 16.0

# Training a logistic regression classifier
logistic = LogisticRegression(max_iter=1000, random_state=42)
logistic.fit(x_train, y_train)

# Hyperparameter tuning for Logistic Regression
hyperlogistic = LogisticRegression(max_iter=1000, random_state=42)
param_grid_logistic = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}
grid_search_logistic = GridSearchCV(hyperlogistic, param_grid_logistic, cv=5, scoring='accuracy')
grid_search_logistic.fit(x_train, y_train)

best_logistic = grid_search_logistic.best_estimator_
print("Best logistic regression hyperparameters:", grid_search_logistic.best_params_)

# Training a support vector machine classifier
svm = SVC(kernel='rbf', gamma='scale', random_state=42)
svm.fit(x_train, y_train)

# Hyperparameter tuning for SVC
hypersvm = SVC(random_state=42)
param_grid_svm = {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto']}
grid_search_svm = GridSearchCV(hypersvm, param_grid_svm, cv=5, scoring='accuracy')
grid_search_svm.fit(x_train, y_train)

best_svm = grid_search_svm.best_estimator_
print("Best SVM hyperparameters:", grid_search_svm.best_params_)


# Train a KNN classifier
knn = KNeighborsClassifier()
knn.fit(x_train, y_train)

# Hyperparameter tuning for KNN Classifier
hyperknn = KNeighborsClassifier()
param_grid_knn = {'n_neighbors': [1, 3, 5, 7, 9]}
grid_search_knn = GridSearchCV(hyperknn, param_grid_knn, cv=5, scoring='accuracy')
grid_search_knn.fit(x_train, y_train)

best_knn = grid_search_knn.best_estimator_
print("Best KNN hyperparameters:", grid_search_knn.best_params_)
print("\n")

# Evaluate models on test set
models = [best_logistic, best_svm, best_knn]
model_names = ['Logistic Regression', 'SVM', 'KNN']
test_accuracies_hyper = []

for model, name in zip(models, model_names):
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    test_accuracies_hyper.append(accuracy)
    print(f"{name} Test Accuracy with Hyperparameter: {accuracy}")
print("\n")

# Plot test accuracies
plt.barh(model_names, test_accuracies_hyper)
plt.title('Test Accuracy Scores with Hyperparameter for Different Models')
plt.xlabel('Accuracy')
plt.show()

models = [logistic, svm, knn]
model_names = [ 'Logistic Regression', 'SVM', 'KNN']
test_accuracies = []

for model, name in zip(models, model_names):
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    test_accuracies.append(accuracy)
    print(f"{name} Test Accuracy without Hyperparameter: {accuracy}")

linear = LinearRegression()
linear.fit(x_train, y_train)
y_pred_linear = np.round(linear.predict(x_test))
accuracy_linear = accuracy_score(y_test, y_pred_linear)
test_accuracies.append(accuracy_linear)
model_names.append('Linear Regression')
print('Linear Regression Test Accuracy without Hyperparameter:', accuracy_linear)
print("\n")

# Plot test accuracies
plt.barh(model_names, test_accuracies)
plt.title('Test Accuracy Scores without Hyperparameter for Different Models')
plt.xlabel('Accuracy')
plt.show()

model_names = [ 'Logistic Regression', 'SVM', 'KNN']

linear = LinearRegression()
linear.fit(x_train, y_train)

models.append(linear)
model_names.append('Linear Regression')

# Evaluate models on validation set
val_accuracies = []

for model, name in zip(models, model_names):
    if name == 'Linear Regression':
        y_pred_val = np.round(model.predict(x_val))
        accuracy_val = accuracy_score(y_val, y_pred_val)
    else:
        y_pred_val = model.predict(x_val)
        accuracy_val = accuracy_score(y_val, y_pred_val)
    
    val_accuracies.append(accuracy_val)
    print(f"{name} Validation Accuracy: {accuracy_val}")

print("\n")

# Plot validation accuracies
plt.barh(model_names, val_accuracies)
plt.title('Validation Accuracy Scores for Different Models')
plt.xlabel('Accuracy')
plt.show()

# Calculate training loss (log loss) for Logistic Regression
y_train_proba_logistic = best_logistic.predict_proba(x_train)
train_loss_logistic = log_loss(y_train, y_train_proba_logistic)
print("Logistic Regression Training Loss:", train_loss_logistic)

# Calculate training loss (MSE) for Linear Regression
y_train_pred_linear = linear.predict(x_train)
train_loss_linear = mean_squared_error(y_train, y_train_pred_linear)
print("Linear Regression Training Loss:", train_loss_linear)

# Plot training losses for Logistic Regression and Linear Regression
training_losses = [train_loss_logistic, train_loss_linear]
training_loss_names = ['Logistic Regression', 'Linear Regression']
plt.barh(training_loss_names, training_losses)
plt.title('Training Loss Scores for Different Models')
plt.xlabel('Loss')
plt.show()

