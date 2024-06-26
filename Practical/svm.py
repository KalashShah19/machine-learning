# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('dataset.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# print("X")
# print(X)
# print("y")
# print(y)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5, random_state = 0)

# print("Training data")
# print(X_train)
# print(y_train)
# print("Test data")
# print(X_test)
# print(y_test)

# Check the unique classes in the training set
# print("Unique classes in y_train:", len(np.unique(y_train)))

# Ensure that y_train contains more than one class
if len(np.unique(y_train)) > 1:
    # Feature Scaling
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # print("X_train")
    # print(X_train)
    # print("X_test")
    # print(X_test)

    # Training the SVM model on the Training set
    from sklearn.svm import SVC
    classifier = SVC(kernel = 'linear', random_state = 0)
    classifier.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    print("Prediction =========================================")
    print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

    # Making the Confusion Matrix
    from sklearn.metrics import confusion_matrix, accuracy_score
    cm = confusion_matrix(y_test, y_pred)
    print("confusion matrix ===========================")
    print(cm)
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

else:
    print("The training data contains only one class. SVM requires at least two classes.")
