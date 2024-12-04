import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
import warnings

warnings.filterwarnings('ignore')

data = pd.read_csv('seeds.csv')

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
y = y - 1

# print(X)

# Find and Print Correlations
print()
print("Correlation of Features with Class")
correlation_with_class = data.corr()['Class'].sort_values(ascending=False)
print(correlation_with_class)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X[:, :2], y, test_size=0.25, random_state=42)

# Train the SVM model
svm_model = SVC(kernel='rbf', gamma='scale', random_state=42)
svm_model.fit(X_train, y_train)

# Make predictions for SVM
y_pred_svm = svm_model.predict(X_test)

# Train the Neural Network (NN) model
nn_model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
nn_model.fit(X_train, y_train)

# Make predictions for NN
y_pred_nn = nn_model.predict(X_test)

svm_score = accuracy_score(y_test, y_pred_svm)
nn_score = accuracy_score(y_test, y_pred_nn)

# Evaluate models
print("\nSVM Classification Report:\n", classification_report(y_test, y_pred_svm))
print("\nNN Classification Report:\n", classification_report(y_test, y_pred_nn))
print(f"Support Vector Machine Accuracy: {svm_score:.2f}")
print(f"Neural Network Accuracy: {nn_score:.2f}")

class_names = ["Kama", "Rosa", "Canadian"]
# Function to plot decision boundaries for both models in the same plot
def plot_decision_boundaries(X, y, model, title):

    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))

    # Predict decision boundary
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot the decision boundary
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.Paired)

    # Plot data points
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', cmap=plt.cm.Paired)
    plt.title(title)
    plt.xlabel("Area")
    plt.ylabel("Perimeter")
    handles, labels = scatter.legend_elements()
    plt.legend(handles, class_names, title="Classes", loc='upper left')

# Create a figure with two subplots side by side
plt.figure(figsize=(14, 6))

# SVM Decision Boundary
plt.subplot(1, 2, 1)
plot_decision_boundaries(X_train, y_train, svm_model, "Support Vector Machine")

# Neural Network Decision Boundary
plt.subplot(1, 2, 2)
plot_decision_boundaries(X_train, y_train, nn_model, "Neural Network")

# Show the plots
plt.show()

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred_svm)

# Display confusion matrix with class names
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap=plt.cm.Blues)
plt.show()