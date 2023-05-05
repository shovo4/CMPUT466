import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, log_loss
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

# Load the digits dataset
digits = load_digits()

# Split the data into training, validation, and testing sets
x_train, x_val, y_train, y_val = train_test_split(digits.data, digits.target, test_size=0.4, random_state=42)
x_val, x_test, y_val, y_test = train_test_split(x_val, y_val, test_size=0.5, random_state=42)

# Scale the data to [0, 1]
x_train = x_train / 16.0
x_val = x_val / 16.0
x_test = x_test / 16.0

# Hyperparameter tuning for Logistic Regression
logistic = LogisticRegression(max_iter=1000, random_state=42)
param_grid_logistic = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}
grid_search_logistic = GridSearchCV(logistic, param_grid_logistic, cv=5, scoring='accuracy')
grid_search_logistic.fit(x_train, y_train)

best_logistic = grid_search_logistic.best_estimator_
print("Best logistic regression hyperparameters:", grid_search_logistic.best_params_)

# Hyperparameter tuning for SVC
svm = SVC(random_state=42)
param_grid_svm = {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto']}
grid_search_svm = GridSearchCV(svm, param_grid_svm, cv=5, scoring='accuracy')
grid_search_svm.fit(x_train, y_train)

best_svm = grid_search_svm.best_estimator_
print("Best SVC hyperparameters:", grid_search_svm.best_params_)

# No hyperparameter tuning needed for Decision Tree Classifier
tree = DecisionTreeClassifier(random_state=42)
tree.fit(x_train, y_train)

# Hyperparameter tuning for KNN Classifier
knn = KNeighborsClassifier()
param_grid_knn = {'n_neighbors': [1, 3, 5, 7, 9]}
grid_search_knn = GridSearchCV(knn, param_grid_knn, cv=5, scoring='accuracy')
grid_search_knn.fit(x_train, y_train)

best_knn = grid_search_knn.best_estimator_
print("Best KNN hyperparameters:", grid_search_knn.best_params_)

# Majority class baseline
dummy = DummyClassifier(strategy='most_frequent')
dummy.fit(x_train, y_train)

# Evaluate models on test set
models = [best_logistic, best_svm, tree, best_knn, dummy]
model_names = ['Logistic Regression', 'SVM', 'Decision Tree', 'KNN', 'Majority Class Baseline']
test_accuracies = []

for model, name in zip(models, model_names):
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    test_accuracies.append(accuracy)
    print(f"{name} Test Accuracy: {accuracy}")

# Plot test accuracies
plt.barh(model_names, test_accuracies)
plt.title('Test Accuracy Scores for Different Models')
plt.xlabel('Accuracy')
plt.show()

# Evaluate models on validation set
val_accuracies = []

for model, name in zip(models, model_names):
    y_pred_val = model.predict(x_val)
    accuracy_val = accuracy_score(y_val, y_pred_val)
    val_accuracies.append(accuracy_val)
    print(f"{name} Validation Accuracy: {accuracy_val}")

# Calculate training loss (log loss) for Logistic Regression
y_train_proba_logistic = best_logistic.predict_proba(x_train)
train_loss_logistic = log_loss(y_train, y_train_proba_logistic)
print("Logistic Regression Training Loss:", train_loss_logistic)

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

# Calculate training loss (log loss) for Logistic Regression
y_train_proba_logistic = best_logistic.predict_proba(x_train)
train_loss_logistic = log_loss(y_train, y_train_proba_logistic)
print("Logistic Regression Training Loss:", train_loss_logistic)

# Calculate training loss (MSE) for Linear Regression
y_train_pred_linear = linear.predict(x_train)
train_loss_linear = mean_squared_error(y_train, y_train_pred_linear)
print("Linear Regression Training Loss:", train_loss_linear)

# Calculate training loss (R^2) for Linear Regression
r2_linear = r2_score(y_train, y_train_pred_linear)
print("Linear Regression R^2:", r2_linear)

# Plot validation accuracies
plt.barh(model_names, val_accuracies)
plt.title('Validation Accuracy Scores for Different Models')
plt.xlabel('Accuracy')
plt.show()

# Plot training losses
training_losses = [train_loss_logistic, train_loss_linear]
training_loss_names = ['Logistic Regression', 'Linear Regression']
plt.barh(training_loss_names, training_losses)
plt.title('Training Loss Scores for Different Models')
plt.xlabel('Loss')
plt.show()

# Plot R^2 scores
r2_scores = [r2_linear]
r2_names = ['Linear Regression']
plt.barh(r2_names, r2_scores)
plt.title('R^2 Scores for Different Models')
plt.xlabel('R^2')
plt.show()

# Plot validation accuracies as line graph
plt.plot(model_names, val_accuracies)
plt.title('Validation Accuracy Scores for Different Models')
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.xticks(rotation=90)
plt.show()

# Plot training losses as line graph
train_losses = [train_loss_logistic, 0, 0, 0, 0, train_loss_linear]
plt.plot(model_names, train_losses)
plt.title('Training Loss Scores for Different Models')
plt.xlabel('Model')
plt.ylabel('Loss')
plt.xticks(rotation=90)
plt.show()