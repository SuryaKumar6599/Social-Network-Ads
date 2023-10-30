# Machine Learning Model Comparison and Hyperparameter Tuning

This repository contains a Python script for comparing the performance of several machine learning models (Support Vector Machine, Gradient Boosting, Decision Tree, and Random Forest) using a dataset from 'Social_Network_Ads.csv'. It also includes hyperparameter tuning for the Random Forest model using Grid Search.

## Overview

In this project, we perform the following tasks:

1. Data Preprocessing:
   - Load the dataset from 'Social_Network_Ads.csv'.
   - Extract features (X) and the target variable (y).
   - Split the data into training and testing sets (75% training, 25% testing) with a fixed random seed for reproducibility.

2. Support Vector Machine (SVM):
   - Train an SVM classifier with a radial basis function (RBF) kernel.
   - Make predictions on the testing data.
   - Evaluate the SVM model's performance using a confusion matrix and accuracy score.

3. Gradient Boosting Classifier:
   - Train a Gradient Boosting Classifier with a specified number of estimators.
   - Make predictions on the testing data.
   - Evaluate the model's performance using a confusion matrix and accuracy score.

4. Decision Tree Classifier:
   - Train a Decision Tree Classifier with a specified maximum depth and 'entropy' criterion.
   - Make predictions on the testing data.
   - Evaluate the model's performance using a confusion matrix and accuracy score.

5. Random Forest Classifier:
   - Train a Random Forest Classifier with a specified number of estimators.
   - Make predictions on the testing data.
   - Evaluate the model's performance using a confusion matrix and accuracy score.
   - Calculate and display feature importances for the Random Forest model.

6. Hyperparameter Tuning for Random Forest:
   - Define a parameter grid with different 'max_depth' and 'n_estimators' values.
   - Perform hyperparameter tuning using Grid Search on the Random Forest model.
   - Display the best hyperparameters and their corresponding accuracy score.

7. Visualizations:
   - Visualize the Decision Tree using text representation.
   - Generate a graphical representation of the Decision Tree.
   - Display the Decision Tree feature importances.

## Requirements

- Python 3
- Required Python libraries: numpy, matplotlib, pandas, scikit-learn
