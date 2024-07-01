import pandas as pd

# Load the sales data
data = pd.read_csv('earnings.csv')

# Calculate the total sales
total = data[' Amount'].sum()

# Find the indices of the minimum and maximum sales entries
min_index = data[' Amount'].idxmin()
max_index = data[' Amount'].idxmax()

# Specify the columns you want to print
columns_of_interest = ['Description', ' Amount']  # replace with your actual column names

# Extract the rows with the minimum and maximum sales entries
min_row = data.loc[min_index, columns_of_interest]
max_row = data.loc[max_index, columns_of_interest]

# Print the total sales, and the specific columns for the minimum and maximum sales entries
print(f"Total Sales = {total}")
print("\nColumns of Minimum Sales Entry:")
print(min_row)
print("\nColumns of Maximum Sales Entry:")
print(max_row)

print("\n Type Wise Analysis")
# Group by the 'Type' column and calculate the sum, min, and max for 'Amount'
grouped_data = data.groupby('Type')[' Amount'].agg(['sum', 'min', 'max']).reset_index()

# Print the resulting dataframe
print(grouped_data)