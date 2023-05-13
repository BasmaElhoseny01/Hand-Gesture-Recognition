# This is the main pipeline for training our model

# Step(0) Import the required utilities
from utils import *

# Step(1) Read Images
training_path='../data_split_resize/'
train_images=read_images(training_path)
print("Files loaded")

# Step(2) Preprocess the images

# Step(3) Extract features
X_train = []
Y_train = []

for i in range(6):
    print(i)
    for img in train_images[str(i)]:
            img = resize(img, (64*4, 128*4))
            fd, hog_image = hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True, channel_axis=2)
            X_train.append(fd)
            Y_train.append(i)
print('Features Extracted')


# Step(4) Train our classifier

# Create an SVM classifier with a linear kernel
clf = svm.SVC(kernel='rbf', C=1, degree=3)
# Fit the classifier to the training data
clf.fit(X_train, Y_train)
print('Model Trained')

# Step(5) Save our model
filename = "../test_pipeline/Trained_SVM.joblib"
joblib.dump(clf, filename)
print('Model exported')
print('Done')