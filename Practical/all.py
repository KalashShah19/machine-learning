#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 10:07:53 2024

@author: bmiit
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('/Users/bmiit/Downloads/dataset.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
# print("Feature matrix (X):")
# print(X)
# print("Target variable (y):")
# print(y)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
# print("Training feature set (X_train):")
# print(X_train)
# print("Test feature set (X_test):")
# print(X_test)
# print("Training target set (y_train):")
# print(y_train)
# print("Test target set (y_test):")
# print(y_test)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# print("Scaled training feature set (X_train):")
# print(X_train)
# print("Scaled test feature set (X_test):")
# print(X_test)

# Training the K-NN model on the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=4)
classifier.fit(X_train, y_train)

# Predicting a new result (example prediction)
# Note: Adjust the input features to match the feature set of your dataset
new_data_scaled = sc.transform(X_train)
new_prediction = classifier.predict(new_data_scaled)
print("Prediction for new data point:")
print(new_prediction)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
comparison = np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1)
print("Comparison of predicted and actual test set results:")
print(comparison)

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
#print(cm)
print("-------")
print(f"Accuracy of K-NN: {accuracy_score(y_test, y_pred)}")

print("")
print("=======================================================")
print("")

# Check the unique classes in the target variable
unique_classes = np.unique(y)
print("Unique classes in the target variable (y):", unique_classes)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Check the distribution of classes in training and testing sets
unique_train_classes = np.unique(y_train)
unique_test_classes = np.unique(y_test)
print("Unique classes in the training target set (y_train):", unique_train_classes)
print("Unique classes in the testing target set (y_test):", unique_test_classes)

if len(unique_train_classes) < 2 or len(unique_test_classes) < 2:
    print("Error: The dataset must have at least two classes for SVM classification.")
else:
    # Feature Scaling
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    # print("Scaled training feature set (X_train):")
    # print(X_train)
    # print("Scaled test feature set (X_test):")
    # print(X_test)

    # Training the SVM model on the Training set
    from sklearn.svm import SVC
    classifier = SVC(kernel='linear', random_state=1)
    classifier.fit(X_train, y_train)

    # Predicting a new result (example prediction)
    new_data_scaled = sc.transform(X_train)
    new_prediction = classifier.predict(new_data_scaled)
    print("Prediction for new data point:")
    print(new_prediction)

    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    comparison = np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1)
    print("Comparison of predicted and actual test set results:")
    print(comparison)

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
print('--------')
print(f"Accuracy of SVM: {accuracy_score(y_test, y_pred)}")


print("===========================================================")



if len(unique_train_classes) < 2 or len(unique_test_classes) < 2:
    print("Error: The dataset must have at least two classes for Naive Bayes classification.")
else:
    # Feature Scaling
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    # print("Scaled training feature set (X_train):")
    # print(X_train)
    # print("Scaled test feature set (X_test):")
    # print(X_test)

    # Training the Naive Bayes model on the Training set
    from sklearn.naive_bayes import GaussianNB
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)

    # Predicting a new result (example prediction)
    new_data_scaled = sc.transform(X_train)
    new_prediction = classifier.predict(new_data_scaled)
    print("Prediction for new data point:")
    print(new_prediction)

    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    comparison = np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1)
    print("Comparison of predicted and actual test set results:")
    print(comparison)

   
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(f"Accuracy of Naive_Bayes: {accuracy_score(y_test, y_pred)}")

