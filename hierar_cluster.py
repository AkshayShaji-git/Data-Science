import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.datasets import make_blobs

# Generate Sample Data (2D Points)
X, _ = make_blobs(n_samples=10, centers=3, random_state=42, cluster_std=1.2)

# Step 1: Compute the linkage matrix using complete linkage
Z = linkage(X, method="complete", metric="euclidean")

# Step 2: Plot the dendrogram to visualize clustering hierarchy
plt.figure(figsize=(8, 5))
dendrogram(Z)
plt.title("Hierarchical Clustering (Complete Linkage)")
plt.xlabel("Data Points")
plt.ylabel("Distance")
plt.show()

# Step 3: Cut the dendrogram to form `n_clusters`
n_clusters = 3
clusters = fcluster(Z, n_clusters, criterion='maxclust')

# Step 4: Plot the clustered data
plt.scatter(X[:, 0], X[:, 1], c=clusters, cmap='rainbow', edgecolors='k')
plt.title("Data Clustered using Complete Linkage")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
