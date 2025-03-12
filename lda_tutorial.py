import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#Define class 1 and class 2 data points with different values
X1 = np.array([[2, 3], [3, 6], [4, 5], [5, 8], [6, 4], [3, 3], [7, 2], [4, 3]]) #Class 1
X2 = np.array([[9, 10], [8, 7], [11, 6], [10, 9], [12, 8], [11, 7], [8, 9], [9, 7]]) #Class 2
#Combine datasets and create labels (0 for Class 1, 1 for Class 2)
X = np.vstack((X1, X2))
y = np.array([0] * len(X1) + [1] * len(X2))
#Perform LDA using sklearn
lda = LinearDiscriminantAnalysis()
lda.fit(X, y)
X_proj = lda.transform(X)
#Scatter plot of LDA projection
plt.scatter(X_proj[:len(X1)], np.zeros(len(X1)), color="red", marker="o", label="Class 1")
plt.scatter(X_proj[len(X1):], np.zeros(len(X2)), color="blue", marker="s", label="Class 2")
plt.axvline(x=0, color="black", linestyle="dashed", label="Decision Boundary")
plt.xlabel("Projected Value")
plt.title("LDA Projection of Data")
plt.legend()
plt.show()
#Predict class for a new sample
new_sample = np.array([[6, 6]])
predicted_class = lda.predict(new_sample)
print(f"Predicted class for {new_sample[0]} is {'Class 2' if predicted_class[0] == 1 else 'Class 1'}")
#Compute class means
mu1, mu2 = np.mean(X1, axis=0), np.mean(X2, axis=0)
#Compute within-class scatter matrix
S1 = np.dot((X1 - mu1).T, (X1 - mu1))
S2 = np.dot((X2 - mu2).T, (X2 - mu2))
SW = S1 + S2
#Compute between-class scatter matrix
SB = np.outer((mu1 - mu2), (mu1 - mu2))
#Compute LDA projection vector
w = np.dot(np.linalg.inv(SW), (mu1 - mu2))
#Project data onto LDA vector
X1_proj, X2_proj = np.dot(X1, w), np.dot(X2, w)
#Plot original data
plt.scatter(X1[:, 0], X1[:, 1], label="Class 1", color="red", marker="o")
plt.scatter(X2[:, 0], X2[:, 1], label="Class 2", color="blue", marker="s")
plt.plot([0, w[0] * 10], [0, w[1] * 10], color="black", linestyle="dashed", label="LDA Direction")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.title("LDA Projection Line")
plt.grid()
plt.show()
#Compute decision threshold
z1, z2 = np.dot(mu1, w), np.dot(mu2, w)
z_threshold = (z1 + z2) / 2
#Scatter plot for 1D projection
plt.scatter(X1_proj, np.zeros_like(X1_proj), color="red", marker="o", label="Class 1")
plt.scatter(X2_proj, np.zeros_like(X2_proj), color="blue", marker="s", label="Class 2")
plt.axvline(x=z_threshold, color="black", linestyle="dashed", label="Decision Boundary")
plt.xlabel("Projected Value")
plt.title("LDA Projection and Decision Boundary")
plt.legend()
plt.show()
#Function to predict class based on projection
def predict_lda(x_new, w, z_threshold):
    z_new = np.dot(x_new, w)
    return 0 if z_new > z_threshold else 1

#Example classification
x_new = np.array([6, 6])
predicted_class = predict_lda(x_new, w, z_threshold)
print(f"Projected Value: {np.dot(x_new, w):.3f}")
print(f"Decision Threshold: {z_threshold:.3f}")
print(f"Predicted Class: {'Class 2' if predicted_class == 1 else 'Class 1'}")
