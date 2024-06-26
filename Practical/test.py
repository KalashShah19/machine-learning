import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

# Load the sales data
data = pd.read_csv('sales.csv')

# Create categories for Total_Sales
bins = [0, 3000, 6000, np.inf]
labels = ['Low', 'Medium', 'High']
data['Sales_Category'] = pd.cut(data['Total_Sales'], bins=bins, labels=labels)

# Define features and target variable
X = data[['Quantity_Sold', 'Price_Per_Unit']]
y = data['Sales_Category']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the SVM classifier
classifier = SVC(kernel='linear', random_state=42)
classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = classifier.predict(X_test)

# Evaluate the model
cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

print(f"Confusion Matrix:\n{cm}")
print(f"Accuracy: {accuracy:.2f}")

# Plotting the confusion matrix
plt.figure(figsize=(8, 6))
plt.matshow(cm, cmap=plt.cm.Blues, fignum=1)
plt.title('Confusion Matrix', pad=20)
plt.colorbar()
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.xticks(ticks=[0, 1, 2], labels=labels)
plt.yticks(ticks=[0, 1, 2], labels=labels)
plt.show()

# # Bar chart for total sales per product
# plt.figure(figsize=(12, 6))
# data.groupby('Product_Name')['Total_Sales'].sum().plot(kind='bar', color='skyblue')
# plt.xlabel('Product Name')
# plt.ylabel('Total Sales')
# plt.title('Total Sales per Product')
# plt.show()

# # Pie chart for sales distribution by product
# sales_by_product = data.groupby('Product_Name')['Total_Sales'].sum()
# plt.figure(figsize=(10, 7))
# sales_by_product.plot(kind='pie', autopct='%1.1f%%', startangle=140)
# plt.title('Sales Distribution by Product')
# plt.ylabel('')
# plt.show()

# # Bar chart for total sales per month
# plt.figure(figsize=(12, 6))
# data.groupby('Month')['Total_Sales'].sum().plot(kind='bar', color='lightgreen')
# plt.xlabel('Month')
# plt.ylabel('Total Sales')
# plt.title('Total Sales per Month')
# plt.show()

# # Pie chart for sales distribution by month
# sales_by_month = data.groupby('Month')['Total_Sales'].sum()
# plt.figure(figsize=(10, 7))
# sales_by_month.plot(kind='pie', autopct='%1.1f%%', startangle=140)
# plt.title('Sales Distribution by Month')
# plt.ylabel('')
# plt.show()
