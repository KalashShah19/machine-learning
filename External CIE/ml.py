import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Step 1: Load the dataset
data = pd.read_csv('seeds.csv')

# Step 2: Calculate correlation with 'Class' column
correlation_with_class = data.corr()['Class'].sort_values(ascending=False)

# Step 3: Display the correlation values
print(correlation_with_class)

# Step 4: Select the features for training (only 'Area' and 'Perimeter' in this case)
X = data[['Area', 'Perimeter']]
y = data['Class']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 5: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 6: SVM Model
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)

# Predict and evaluate the SVM model
svm_predictions = svm_model.predict(X_test)
svm_accuracy = accuracy_score(y_test, svm_predictions)
print(f"SVM Accuracy: {svm_accuracy:.2f}")

# Step 7: Neural Network Model
nn_model = MLPClassifier(hidden_layer_sizes=(50,), max_iter=1000, random_state=42)
nn_model.fit(X_train, y_train)

# Predict and evaluate the NN model
nn_predictions = nn_model.predict(X_test)
nn_accuracy = accuracy_score(y_test, nn_predictions)
print(f"NN Accuracy: {nn_accuracy:.2f}")

# Step 8: Visualize the results
# We'll plot the results as a 2D decision boundary, using the same 'Area' and 'Perimeter' features
plt.figure(figsize=(12, 6))

# Step 9: SVM decision boundary
plt.subplot(1, 2, 1)
h = .02  # Step size in the mesh
x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

Z = svm_model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.8)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, edgecolors='k', marker='o', s=100)
plt.title('SVM Decision Boundary')

# Step 10: Neural Network decision boundary
plt.subplot(1, 2, 2)
Z_nn = nn_model.predict(np.c_[xx.ravel(), yy.ravel()])
Z_nn = Z_nn.reshape(xx.shape)

plt.contourf(xx, yy, Z_nn, alpha=0.8)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, edgecolors='k', marker='o', s=100)
plt.title('NN Decision Boundary')

plt.show()
