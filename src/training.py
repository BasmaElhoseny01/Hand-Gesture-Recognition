from utils import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import time

# Generate a random dataset with 100000 samples and 20 features
X, y = make_classification(n_samples=1000000, n_features=20, random_state=42)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Random Forest classifier with 100 trees
rf = RandomForestClassifier(n_estimators=100, random_state=42, criterion='log_loss')

start_time = round(time.time() * 1000)
# Fit the classifier to the training data
rf.fit(X_train, y_train)
end_time = round(time.time() * 1000)
print("Training time = ", end_time - start_time)

start_time = round(time.time() * 1000)
# Predict the labels of the test data
y_pred = rf.predict(X_test)
end_time = round(time.time() * 1000)
print("Prediction time = ", end_time - start_time)

# Evaluate the accuracy of the classifier
accuracy = rf.score(X_test, y_test)
print("Accuracy:", accuracy)

def train():
    return None


# Random Forest
# Naive Bayes
# SVM
# Logistic Regression
# K Nearest Neighbor
# Gradient Boosting Machines