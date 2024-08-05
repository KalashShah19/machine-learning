import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("dataset.csv")

# Create a DataFrame
df = pd.DataFrame(data)

# Drop the ID column for clustering
X = df.drop(columns=['CUST_ID'])

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform hierarchical clustering
Z = linkage(X_scaled, method='ward')

# Create clusters
num_clusters = 3  # Define the number of clusters
df['Cluster'] = fcluster(Z, num_clusters, criterion='maxclust')

# Print the number of samples in each cluster
print("\nCluster Counts:\n", df['Cluster'].value_counts())

# Plot the dendrogram
plt.figure(figsize=(10, 7))
dendrogram(Z, truncate_mode='lastp', p=30, leaf_rotation=90)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.show()

# Plot clusters in a 2D space using the first two features
plt.figure(figsize=(10, 8))
sns.scatterplot(x='Feature1', y='Feature2', hue='Cluster', data=df, palette='viridis', s=100)
plt.title('Hierarchical Clustering Results')
plt.xlabel('Feature1')
plt.ylabel('Feature2')
plt.legend(title='Cluster')
plt.show()