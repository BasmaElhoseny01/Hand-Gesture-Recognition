from utils import *


def extract_features():
    return None


# ORB
# SIFT
# SURF


# CORNER DETECTION 
def detectCorners(filename):
    # read the image
    img = cv2.imread(filename)
    # plt.imshow(img), plt.show()

    # img = cv2.dilate(img, (50, 20), img)
    # plt.imshow(img), plt.show()

    # convert image to gray scale image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # detect corners with the goodFeaturesToTrack function.
    corners = cv2.goodFeaturesToTrack(gray, 0, 0.8, 10)
    corners = np.int0(corners)
    # print(corners[0][0])
    corners_detected = []
    # we iterate through each corner, 
    # making a circle at each point that we think is a corner.
    for corner in corners:
        x, y = corner.ravel()
        # print(x, y)
        corners_detected.append([x, y])
        centre = (x, y)
        color = (255, 0, 0)
        radius = 50
        thickness = 10

        img = cv2.circle(img, centre, radius, color, thickness)
    
    corners_detected = np.asarray(corners_detected)

    # print(corners_detected)
    
    # plt.imshow(img), plt.show()

    return corners_detected