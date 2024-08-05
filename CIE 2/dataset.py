import pandas as pd

# Define the path to your input and output CSV files
input_csv_path = 'CC GENERAL.csv'  # Replace with your input file path
output_csv_path = 'dataset.csv'  # Replace with your desired output file path

# Read the CSV file
df = pd.read_csv(input_csv_path)

# Select the first 25 rows
df_subset = df.head(31)

# Write the subset to a new CSV file
df_subset.to_csv(output_csv_path, index=False)

print(f"First 30 rows have been written to {output_csv_path}")
