import numpy as np 
from sklearn.tree import DecisionTreeRegressor  
from sklearn.datasets import make_regression 
from sklearn.model_selection import train_test_split 

class RandomForestRegressorCustom:
    def __init__(self, n_trees=10, max_depth=None):
      
        self.n_trees = n_trees  # Store number of trees
        self.max_depth = max_depth  # Store maximum depth of trees
        self.trees = []  # List to store trained decision trees

    def _bootstrap_sample(self, X, y):
       
        n_samples = X.shape[0]  # Number of samples in the dataset
        idxs = np.random.choice(n_samples, n_samples, replace=True)  # Randomly select indices with replacement
        return X[idxs], y[idxs]  # Return sampled features and target values

    def fit(self, X, y):
        self.trees = []  # Clear previous trees (if any)
        for _ in range(self.n_trees):  # Iterate for the number of trees
            X_sample, y_sample = self._bootstrap_sample(X, y)  # Get a bootstrapped sample
            tree = DecisionTreeRegressor(max_depth=self.max_depth)  # Create a new decision tree with given max depth
            tree.fit(X_sample, y_sample)  # Train the tree on the bootstrapped dataset
            self.trees.append(tree)  # Store the trained tree

    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])  # Get predictions from all trees
        return np.mean(predictions, axis=0)  # Compute average prediction (final output)

# Create a synthetic regression dataset
X, y = make_regression(n_samples=200, n_features=5, noise=0.1)  # 200 samples, 5 features, with some noise

# Split dataset into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize Random Forest model with 10 trees, each having max depth 5
rf = RandomForestRegressorCustom(n_trees=10, max_depth=5)
rf.fit(X_train, y_train)  # Train the model on training data

# Predict outputs for test data
y_pred = rf.predict(X_test)

# Display first 5 predicted values
print("Sample Predictions:", y_pred[:5])
