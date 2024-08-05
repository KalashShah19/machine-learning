import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

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

# Define a function to plot circles around clusters
def plot_cluster_circles(ax, df, feature_x, feature_y, cluster_col, palette):
    unique_clusters = df[cluster_col].unique()
    for cluster in unique_clusters:
        cluster_data = df[df[cluster_col] == cluster]
        x_mean = cluster_data[feature_x].mean()
        y_mean = cluster_data[feature_y].mean()
        x_std = cluster_data[feature_x].std()
        y_std = cluster_data[feature_y].std()
        max_std = max(x_std, y_std)
        
        # Plot scatter points
        sns.scatterplot(x=cluster_data[feature_x], y=cluster_data[feature_y], 
                        hue=df[cluster_col], palette=palette, ax=ax, 
                        edgecolor='w', s=100, alpha=0.6, legend=False)
        
        # Draw a circle around the cluster
        # Ensure we do not access an out-of-range index
        circle_color = palette[cluster % len(palette)]  # Use modulo to ensure valid index
        circle = plt.Circle((x_mean, y_mean), 2 * max_std, color=circle_color, fill=False, linestyle='--')
        ax.add_patch(circle)

# Plotting both outputs in separate halves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

# --- Hierarchical Clustering Half ---

# Plot the dendrogram on the first subplot
dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=False, ax=ax1)
ax1.set_title('Hierarchical Clustering Dendrogram')
ax1.set_xlabel('Sample index')
ax1.set_ylabel('Distance')

# Scatter plot with circles for Hierarchical Clustering
palette_hc = sns.color_palette('coolwarm', n_colors=n_clusters_hc)
plot_cluster_circles(ax2, df_numeric, 'BALANCE', 'PURCHASES', 'Hierarchical_Cluster', palette_hc)

# --- K-means Clustering Half ---

# Scatter plot with circles for K-means Clustering on the second subplot
palette_kmeans = sns.color_palette('viridis', n_colors=n_clusters_kmeans)
plot_cluster_circles(ax2, df_numeric, 'BALANCE', 'PURCHASES', 'KMeans_Cluster', palette_kmeans)

# Set labels and titles for the scatter plot
ax2.set_title('Clustering Results')
ax2.set_xlabel('Balance')
ax2.set_ylabel('Purchases')

# Combine legends and remove duplicate labels
handles, labels = ax2.get_legend_handles_labels()
unique_labels = list(dict.fromkeys(labels))  # Remove duplicates
unique_handles = [handles[labels.index(label)] for label in unique_labels]
ax2.legend(unique_handles, unique_labels, loc='upper right', title='Clusters')

plt.tight_layout()
plt.show()