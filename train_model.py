from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

# Load the Iris dataset
X, y = load_iris(return_X_y=True)

# Create and train the model
clf = LogisticRegression(max_iter=200)
clf.fit(X, y)

# Confirmation
print("Model trained")
