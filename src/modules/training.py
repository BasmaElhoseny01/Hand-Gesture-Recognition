from utils import *

def train_model(X_train,Y_train,option):
    clf=None
    if(option=="svm"):
        # Create an SVM classifier with a linear kernel
        clf = svm.SVC(kernel='rbf', C=10, degree=3)
        # Fit the classifier to the training data
        clf.fit(X_train, Y_train)
        print('SVM Model Trained')

    elif(option=="rf"):
        # Create a Random Forest classifier with 100 trees
        clf= RandomForestClassifier(n_estimators=100, random_state=42, criterion='log_loss')

        # Fit the classifier to the training data
        clf.fit(X_train, Y_train)
        print('RF Model Trained')

    else:
        print("Wrong Model Option!!!",option)
        raise TypeError("Wrong Model Option")

    return clf



def train_randomforest(X_train,y_train,X_test,y_test):
    # Split the dataset into training and testing sets
    # X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Create a Random Forest classifier with 100 trees
    rf = RandomForestClassifier(n_estimators=100, random_state=42, criterion='log_loss')

    # Fit the classifier to the training data
    rf.fit(X_train, y_train)

    # Predict the labels of the test data
    y_pred = rf.predict(X_test)

    return y_pred, y_test

def train_svm(X_train,y_train,X_test,y_test):
    # Split the dataset into training and testing sets
    # X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Create an SVM classifier with a linear kernel
    clf = svm.SVC(kernel='linear')

    # Fit the classifier to the training data
    clf.fit(X_train, y_train)

    # Predict the labels of the test data
    y_pred = clf.predict(X_test)

    return y_pred, y_test

# Random Forest
# Naive Bayes
# SVM
# Logistic Regression
# K Nearest Neighbor
# Gradient Boosting Machines