import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# Load the Seeds dataset (downloaded from UCI repository)
# url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00236/seeds_dataset.txt"
# column_names = ['Area', 'Perimeter', 'Compactness', 'Kernel_Length', 'Kernel_Width', 
#                 'Asymmetry_Coefficient', 'Kernel_Groove_Length', 'Class']
# data = pd.read_csv(url, delim_whitespace=True, names=column_names)
data = pd.read_csv('seeds.csv')

# Select features and target
X = data.iloc[:, :-1].values  # Use the first 3 features
y = data.iloc[:, -1].values  # Class labels (1, 2, 3)
print(X)
# Map class labels to 0, 1, 2 for compatibility with scikit-learn
y = y - 1

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X[:, :2], y, test_size=0.3, random_state=42)

# Train the SVM model
svm_model = SVC(kernel='rbf', gamma='scale', random_state=42)
svm_model.fit(X_train, y_train)

# Make predictions
y_pred = svm_model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Plot decision boundaries
def plot_decision_boundaries(X, y, model):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.Paired)
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', cmap=plt.cm.Paired)
    plt.title("SVM Decision Boundaries (Seeds Dataset)")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend(*scatter.legend_elements(), title="Classes")
    plt.show()

# Plot decision boundaries
plot_decision_boundaries(X_train, y_train, svm_model)

