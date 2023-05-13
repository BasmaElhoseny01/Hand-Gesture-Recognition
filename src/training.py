from utils import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def train_randomforest(X_train,y_train,X_test,y_test):
    # Split the dataset into training and testing sets
    # X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Create a Random Forest classifier with 100 trees
    rf = RandomForestClassifier(n_estimators=100, random_state=42, criterion='log_loss')

    # Fit the classifier to the training data
    rf.fit(X_train, y_train)

    # Predict the labels of the test data
    y_pred = rf.predict(X_test)

    return y_pred, y_test,rf

def train_svm(X_train,y_train,X_test,y_test):
    # Split the dataset into training and testing sets
    # X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Create an SVM classifier with a linear kernel
    clf = svm.SVC(kernel='linear')

    # Fit the classifier to the training data
    clf.fit(X_train, y_train)

    # Predict the labels of the test data
    y_pred = clf.predict(X_test)

    return y_pred, y_test,clf

# Random Forest
# Naive Bayes
# SVM
# Logistic Regression
# K Nearest Neighbor
# Gradient Boosting Machines