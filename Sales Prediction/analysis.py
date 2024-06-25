import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from io import StringIO

# Load the sample sales data
data = pd.read_csv("sales.csv")
data['Date'] = pd.to_datetime(data[['Year', 'Month']].assign(DAY=1))

# Display the data to check if it loaded correctly
print(data)

# Define the features (Quantity_Sold and Price_Per_Unit) and the target (Total_Sales)
X = data[['Quantity_Sold', 'Price_Per_Unit']]
y = data['Total_Sales']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print the model coefficients
print("Coefficients:")
print(f"Intercept: {model.intercept_}")
print(f"Quantity_Sold coefficient: {model.coef_[0]}")
print(f"Price_Per_Unit coefficient: {model.coef_[1]}")

# Calculate and print additional statistics
total_revenue = data['Total_Sales'].sum()
total_units_sold = data['Quantity_Sold'].sum()
min_quantity_sold = data['Quantity_Sold'].min()
max_quantity_sold = data['Quantity_Sold'].max()

print(f"Total Revenue: {total_revenue} Rs")
print(f"Total Units Sold: {total_units_sold}")
print(f"Minimum Quantity Sold: {min_quantity_sold}")
print(f"Maximum Quantity Sold: {max_quantity_sold}")

# Find the month with maximum and minimum sales
monthly_sales = data.groupby(['Year', 'Month'])['Total_Sales'].sum()
max_sales_month = monthly_sales.idxmax()
min_sales_month = monthly_sales.idxmin()

print(f"Month with Maximum Sales: {max_sales_month}")
print(f"Month with Minimum Sales: {min_sales_month}")

# Find the product with maximum and minimum sales
product_sales = data.groupby('Product_Name')['Total_Sales'].sum()
max_sales_product = product_sales.idxmax()
min_sales_product = product_sales.idxmin()

print(f"Product with Maximum Sales: {max_sales_product}")
print(f"Product with Minimum Sales: {min_sales_product}")

# Plotting

# Bar chart of monthly sales
plt.figure(figsize=(10, 6))
monthly_sales.plot(kind='bar', color='skyblue')
plt.title('Monthly Sales')
plt.xlabel('Month')
plt.ylabel('Total Sales (Rs)')
plt.xticks(rotation=45)
plt.show()

# Pie chart of sales by product
plt.figure(figsize=(8, 8))
product_sales.plot(kind='pie', autopct='%1.1f%%', startangle=140)
plt.title('Sales Distribution by Product')
plt.ylabel('')
plt.show()

# Scatter plot of Quantity_Sold vs Total_Sales
plt.figure(figsize=(14, 5))

plt.subplot(1, 3, 1)
plt.scatter(data['Quantity_Sold'], data['Total_Sales'], color='blue')
plt.xlabel('Quantity Sold')
plt.ylabel('Total Sales')
plt.title('Quantity Sold vs Total Sales')

# Scatter plot of Price_Per_Unit vs Total_Sales
plt.subplot(1, 3, 2)
plt.scatter(data['Price_Per_Unit'], data['Total_Sales'], color='green')
plt.xlabel('Price Per Unit')
plt.ylabel('Total Sales')
plt.title('Price Per Unit vs Total Sales')

# Scatter plot of Actual vs Predicted Total_Sales
plt.subplot(1, 3, 3)
plt.scatter(y_test, y_pred, color='red')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.xlabel('Actual Total Sales')
plt.ylabel('Predicted Total Sales')
plt.title('Actual vs Predicted Total Sales')

plt.tight_layout()
plt.show()