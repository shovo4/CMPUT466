# CMPUT466

## Coding Assignment 1 (Linear Regression)

- implementing the closed-form solution to linear regression and it will use matrix-vector representations for data representation and get some experience with data visualization. In addition, it will explore the phenomenon of “regression to the mean.” 

- generating data with 2 steps
    - Independently generate α ~ N(0,σ_a^2) and b ~ N(0,σ_b^2)
    - Rotate (a, b)^T to a degree (eg 45°) and obtain (x,y)^T
- In the function leastSquares(X, Y), X is simply the input and Y is simply the output. It is not related to x2y regression or y2x regression.
- In second problem, we will explore gradient descent optimization for linear regression, applied to the Boston house price prediction
- Training loss (MSE)
- The measure of success
- hyperparameters

## Coding Assignment 2 (Linear regression and Logistic Regression)

- binary classification problem
-  In particular, this problem shows that linear regression is a bad model for classification problems.
- Two Datasets for each dataset we have 400 samples in total, where 200 are positive and 200 are negative. The dataset is plotted in the left panel below. 
- Train a classifier by thresholding a linear regression model and logistic regression classifier 
- stochastic gradient descent method (SGD)
- Training Accuracy for linear and logistic regression
- The number of epoch that yields the best validation performance,
- The validation performance (accuracy) in that epoch, and
- The test performance (accuracy) in that epoch. 
- plots: 
    - The learning curve of the training cross-entropy loss, and
    - The learning curve of the validation accuracy.

## Mini - Project

- The basic goal of the mini-project is for the student to gain first-hand experience in formulating a task as a machine learning problem and have a rigorous practice of applying machine learning algorithms. 
- Digit Dataset
    - The digits dataset is a set of 8x8 images of handwritten digits ranging from 0 to 9
    - The objective of the models is to correctly classify the digit in each image.
- Four different models were tested on the Digits dataset, including Logistic Regression (multiclass classifier), SVM,  KNN, and Linear Regression.
- The data was split into training, validation and test sets. 60% of the data used for training and then 20% of the data used for validation. At last, 20% of the data used for testing
- Next, the data is scaled to the range [0, 1] using division by 16.0. This is done to ensure that all features have similar ranges, which can help improve model performance.
- The code then trains three models, Logistic Regression, Support Vector Machine (SVM), and K-Nearest Neighbors (KNN), on the training set using the default hyperparameters. After training the models, they are hyperparameter tuned using a grid search approach, which tries different combinations of hyperparameters from a predefined list of values. The best hyperparameters for each model are chosen based on the accuracy score on the training set. The hyperparameters tuning process is done using the GridSearchCV function from scikit-learn's model_selection module, and the accuracy metric is used for scoring.
- Then, the code evaluates the models' performance on the test set using the best hyperparameters obtained from the grid search. The accuracy score is computed for each model using scikit-learn's accuracy_score function. A bar graph is plotted using Matplotlib to compare the models' performance. The same process is repeated for the models without hyperparameter tuning, and the results are plotted to compare the performance with and without hyperparameter tuning.
- The code also evaluates the models' performance on the validation set. This is done to see how the models generalize to unseen data. The accuracy score is computed using scikit-learn's accuracy_score function, and the results are plotted and compared using Matplotlib.
- Finally, the code calculates the training loss for the logistic regression (log_loss) and linear regression models (mean_squared_error). The results are compared by using plots

- Results
- KNN and SVM models were the most accurate models for the Digits dataset. 
- Linear Regression performed poorly, as it is not suitable for classification tasks.


