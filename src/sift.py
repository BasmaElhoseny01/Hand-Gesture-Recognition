# IMPORTS
import cv2
import utils

# Loading the image
img = cv2.imread('two_1.jpg')
 
 # Converting image to grayscale
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# Applying SIFT detector
sift = cv2.xfeatures2d.SIFT_create()
kp = sift.detect(gray, None)
 
# Marking the keypoint on the image using circles
img = cv2.drawKeypoints(gray, kp, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

utils.show_images([img])

