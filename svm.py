import numpy as np  

class SVM:
    def __init__(self, lr=0.01, lambda_param=0.01, n_iters=1000):
        
        self.lr = lr  
        self.lambda_param = lambda_param  
        self.n_iters = n_iters  
        self.w = None  # Weight vector
        self.b = None  # Bias term

    def fit(self, X, y):
        n_samples, n_features = X.shape  
        self.w = np.zeros(n_features)  # Initialize weights as zeros
        self.b = 0  # Initialize bias as zero

        y = np.where(y <= 0, -1, 1)  # Ensure labels are -1 or 1

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):  
                condition = y[idx] * (np.dot(x_i, self.w) + self.b) >= 1  

                if condition:
                    # No misclassification: Apply only weight decay (L2 regularization)
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    # Misclassification: Update both w and b
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y[idx]))
                    self.b -= self.lr * y[idx]

    def predict(self, X):
       
        linear_output = np.dot(X, self.w) + self.b  
        return np.sign(linear_output)  # Convert to -1 or 1

# Example Usage:
if __name__ == "__main__":
    # Generate simple dataset (binary classification)
    X_train = np.array([[2, 3], [1, 1], [2, 1], [3, 3], [2, 4], [4, 1]])
    y_train = np.array([1, -1, -1, 1, 1, -1])  # Labels must be -1 or 1

    # Train SVM
    model = SVM(lr=0.01, lambda_param=0.01, n_iters=1000)
    model.fit(X_train, y_train)

    # Predict on some points
    X_test = np.array([[2, 2], [3, 2], [4, 4]])
    predictions = model.predict(X_test)

    print("Predictions:", predictions)
