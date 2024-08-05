import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# Load the dataset
input_csv_path = 'dataset.csv'  # Replace with your file path
df = pd.read_csv(input_csv_path)

# Drop the non-numeric column 'CUST_ID' for clustering
df_numeric = df.drop(columns=['CUST_ID'], errors='ignore')

# Check for missing values and handle them
print("Missing values before handling:\n", df_numeric.isnull().sum())
df_numeric = df_numeric.dropna()  # Alternatively, use imputation

# Verify that the number of rows is consistent
print(f"Rows in original DataFrame: {df.shape[0]}")
print(f"Rows in cleaned DataFrame: {df_numeric.shape[0]}")

# Standardize the features
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_numeric)

# Apply K-means clustering
kmeans = KMeans(n_clusters=4, random_state=42)  # Adjust the number of clusters as needed
clusters = kmeans.fit_predict(df_scaled)

# Check the length of clusters
print(f"Number of clusters: {len(clusters)}")
print(f"Length of DataFrame index: {df_numeric.shape[0]}")

# Assign cluster labels to the cleaned DataFrame
df_numeric['Cluster'] = clusters

# If the cleaned DataFrame is not the same length as the clusters, it will raise an error.
# Ensure that they are of the same length
assert len(df_numeric) == len(clusters), "Length mismatch between DataFrame and clusters"

# Print the cluster centers
print("Cluster Centers:\n", kmeans.cluster_centers_)

# Print the number of samples in each cluster
print("\nCluster Counts:\n", df_numeric['Cluster'].value_counts())

# Visualize the clusters using the first two features for simplicity
plt.figure(figsize=(12, 8))
sns.scatterplot(x=df_numeric['BALANCE'], y=df_numeric['PURCHASES'], hue=df_numeric['Cluster'], palette='viridis', s=100)
plt.title('Clustering of Credit Card Customers')
plt.xlabel('Balance')
plt.ylabel('Purchases')
plt.legend(title='Cluster')
plt.show()

# Reduce dimensions to 2D for visualization
pca = PCA(n_components=2)
df_pca = pca.fit_transform(df_scaled)

# Create a DataFrame for PCA results
df_pca_df = pd.DataFrame(df_pca, columns=['PC1', 'PC2'])
df_pca_df['Cluster'] = df_numeric['Cluster']

# Plot PCA results
plt.figure(figsize=(12, 8))
sns.scatterplot(x='PC1', y='PC2', hue='Cluster', data=df_pca_df, palette='viridis', s=100)
plt.title('PCA of Credit Card Customer Clusters')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Cluster')
plt.show()
