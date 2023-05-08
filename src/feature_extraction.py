from utils import *


def extract_features(path_of_the_directory, classification, debug=False):
    feature_descriptor = []     # size: i * n * 128 , i is the number of images, n is the number of key points
    expected_class = []         # List of len = i, i is the number of images

    for filename in os.listdir(path_of_the_directory):
        img = cv2.imread(path_of_the_directory+str(filename), cv2.IMREAD_GRAYSCALE)
        corners_descriptors=Harris(img, debug)
        feature_descriptor.append(corners_descriptors)
        expected_class.append(classification)

    return feature_descriptor, expected_class

# ORB
# SIFT
# SURF

def cluster_descriptors(feature_descriptor, expected, num_of_clusters=3):
    #Get the descriptors of the found key points in the images
    descriptors = [j for k in feature_descriptor if k is not None for j in k]   #Cleaner version
    num_of_images = len(feature_descriptor)
    kmeans = KMeans(n_clusters=num_of_clusters).fit(descriptors)
    final_fv = np.zeros((num_of_images, num_of_clusters))
    k_indx = 0
    for img_indx in range(len(feature_descriptor)):
        if feature_descriptor[img_indx] is not None:
            for p in range(len(feature_descriptor[img_indx])):
                final_fv[img_indx][kmeans.labels_[k_indx]] += 1
                k_indx += 1
    return final_fv

def Harris(img, debug=False):
    '''
    img:"grey"
    '''
    block_size = 2
    aperture_size = 3
    k = 0.04
    harris_img = cv2.cornerHarris(img, block_size, aperture_size, k)

    harris_img = cv2.normalize(harris_img, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    # Create SIFT descriptor
    sift = cv2.SIFT_create()

    # Detect and compute SIFT descriptors for Harris corners
    keypoints, descriptors = sift.detectAndCompute(harris_img, None)

    if (debug):
        output_image = cv2.drawKeypoints(img, keypoints, 0, (0, 0, 255),
                                         flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
        show_images([output_image])

    return descriptors

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