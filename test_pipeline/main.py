# This is the main pipeline for testing our model

# Step(0) Import the required utilities
from utils import *

# Step(1) Read Images
path='../data_split_resize/'
test_images=read_images(path,type="test")
print('Images loaded')

# Step(2) Preprocess the images

# Step(3) Extract features
# Note: We need to edit this part in order not to read images based on classes (i.e. one bulk of images)
X_test = []
Y_test = []
for i in range(6):
    print(i)
    for img in test_images[str(i)]:
            fd, hog_image = hog(img, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2), visualize=True, channel_axis=2)
            X_test.append(fd)
            Y_test.append(i)
print('Features Extracted')

test_images = None

# Step(4) Load our model
filename = "Trained_SVM.joblib"
loaded_model = joblib.load(filename)
print('Model loaded')

# Step(5) Evaluate the results
result = loaded_model.predict(X_test)
print('Results evaluated')

# Step(6) Write the results into txt file
y = ["{}\n".format(i) for i in result]
with open('results.txt', 'w') as fp:
    fp.writelines(y)
print('Results saved')

y = ["{}\n".format(i) for i in Y_test]
with open('expected.txt', 'w') as fp:
    fp.writelines(y)
print('Expected outputs saved')