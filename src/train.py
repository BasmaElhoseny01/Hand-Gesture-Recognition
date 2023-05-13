# This is the main pipeline for training our model

# Step(0) Import the required utilities
from utils import *
from modules.preprocessing import preprocessing


# Step(1) Read Images
training_path = '../data_split_resize/'
train_images = read_images(training_path)
print("Files loaded")



# Step(2) Preprocess the images
train_images = preprocessing(train_images)
print('Preprocessing Done')



# Step(3) Extract features
X_train = []
Y_train = []

for i in range(6):
    print(i)
    for img in train_images[str(i)]:
        fd, hog_image = hog(img, orientations=9, pixels_per_cell=(
            16, 16), cells_per_block=(2, 2), visualize=True, channel_axis=2)
        X_train.append(fd)
        Y_train.append(i)
print('Features Extracted')
train_images = None



# Step(4) Train our classifier
# Create an SVM classifier with a linear kernel
clf = svm.SVC(kernel='rbf', C=1, degree=3)
# Fit the classifier to the training data
clf.fit(X_train, Y_train)
print('SVM Model Trained')

# Step(5) Save our model
filename = "../models/Trained_SVM.joblib"
joblib.dump(clf, filename)
print('RF Model exported')

# Step(4) Train our classifier
# Create a Random Forest classifier with 100 trees
clf= RandomForestClassifier(n_estimators=100, random_state=42, criterion='log_loss')

# Fit the classifier to the training data
clf.fit(X_train, Y_train)
print('RF Model Trained')

# Step(5) Save our model
filename = "../models/Trained_RF.joblib"
joblib.dump(clf, filename)
print('RF Model exported')
print('Done Training :D')
