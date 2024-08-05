import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
input_csv_path = 'dataset.csv'  # Your dataset file name
df = pd.read_csv(input_csv_path)

# Drop the non-numeric column 'CUST_ID' for clustering if it exists
df_numeric = df.drop(columns=['CUST_ID'], errors='ignore')

# Check for missing values and handle them
df_numeric = df_numeric.dropna()  # Alternatively, use imputation

# Standardize the features
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_numeric)

# --- K-means Clustering ---

# Apply K-means clustering
n_clusters_kmeans = 4
kmeans = KMeans(n_clusters=n_clusters_kmeans, n_init=10, random_state=42)  # Set n_init explicitly
df_numeric['KMeans_Cluster'] = kmeans.fit_predict(df_scaled)

# --- Hierarchical Clustering ---

# Perform hierarchical clustering
n_clusters_hc = 4
linked = linkage(df_scaled, method='ward')
df_numeric['Hierarchical_Cluster'] = fcluster(linked, n_clusters_hc, criterion='maxclust')  # Adjust number of clusters if needed

# Plotting both outputs in separate halves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

# --- Hierarchical Clustering Half ---

# Plot the dendrogram on the first subplot
dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=False, ax=ax1)
ax1.set_title('Hierarchical Clustering Dendrogram')
ax1.set_xlabel('Sample index')
ax1.set_ylabel('Distance')

# Scatter plot for Hierarchical Clustering on the second subplot
sns.scatterplot(x=df_numeric['BALANCE'], y=df_numeric['PURCHASES'],
                hue=df_numeric['Hierarchical_Cluster'], palette='coolwarm',
                marker='o', s=100, edgecolor='w', ax=ax2, alpha=0.6, legend=False)

# Scatter plot for K-means Clustering on the same subplot
sns.scatterplot(x=df_numeric['BALANCE'], y=df_numeric['PURCHASES'],
                hue=df_numeric['KMeans_Cluster'], palette='viridis',
                marker='o', s=100, edgecolor='w', ax=ax2, alpha=0.6, legend=False)

# Set labels and titles for the scatter plot
ax2.set_title('Clustering Results')
ax2.set_xlabel('Balance')
ax2.set_ylabel('Purchases')

# Combine legends for K-means and Hierarchical Clustering
handles_hc, labels_hc = ax2.get_legend_handles_labels()
unique_labels_hc = list(dict.fromkeys(labels_hc))  # Remove duplicates
unique_handles_hc = [handles_hc[labels_hc.index(label)] for label in unique_labels_hc]

# Add legends for K-means and Hierarchical Clustering
ax2.legend(handles=unique_handles_hc, labels=unique_labels_hc, loc='upper right', title='Clusters')

plt.tight_layout()
plt.show()
