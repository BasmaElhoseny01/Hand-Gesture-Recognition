from utils import *
def extract_features(images,option):
    X = []
    Y = []
    for i in range(6):
        print(i)
        for img in images[str(i)]:
            if(option=="hog"):
                fd, hog_image = hog(img, orientations=9, pixels_per_cell=(
                16, 16), cells_per_block=(2, 2), visualize=True, channel_axis=2)
                X.append(fd)
                Y.append(i)
            else:
                print("Wrong Feature Option!!!",option)
                raise TypeError("Wrong Feature Option")
    return X,Y




def sift_descriptors(images, debug=False):
    """
    Function to return all descriptors for all images according to the descriptor[Feature extractor used] 
    """

    # SIFT Feature Extractor
    sift = cv2.SIFT_create()

    # Dictionary {'0':[[keypts_of_img1*128],[keypts_of_img2*128],[keypts_of_img3*128]..]}
    sift_vectors = {'0':None,'1':None,'2':None,'3':None,'4':None,'5':None}
    descriptor_list = []  # Total number of keypoints for all data set images * 128

    for key, value in images.items():
        # Category by Category 0,1,..5
        if (debug):
            print("sift_descriptors for "+key)

        class_descriptors = []
        i = 0
        for img in value:
            # Loop over Dictionary items value by value M0 M1 M2
            # Every image in the category we are in
            keypoints, descriptors = sift.detectAndCompute(img, None)
            if (debug):
                print(str(i)+" has: " + str(len(keypoints)) + "Keypoints")
                i = i+1

            # Normalize descriptors
            if descriptors is None:
                print("No Key Points Detected")

            descriptor_list.extend(descriptors)
            class_descriptors.append(descriptors)

        # print("shape", np.shape(class_descriptors))
        # print("shape", np.shape(class_descriptors[0]))
        # print("shape", (class_descriptors[0]))

        # print(sift_vectors[key])
        sift_vectors[key]=class_descriptors  # [[keypts_img1*128],[keypts_img2*128],[keypts_img3*128]]
        # print(sift_vectors[key])
    return descriptor_list, sift_vectors

def orb_descriptors(images, params = None, debug=False):
    # initialize ORB detector with custom parameters
    if params is not None:
        orb = cv2.ORB_create(params[0], params[1], params[2])
        #                    nFeatures, scaleFactor, nLevels
    else:
        orb = cv2.ORB_create()
    # Dictionary {'0':[[keypts_of_img1*128],[keypts_of_img2*128],[keypts_of_img3*128]..]}
    orb_vectors = {'0':None,'1':None,'2':None,'3':None,'4':None,'5':None}
    descriptor_list = []  # Total number of keypoints for all data set images * 128

    for key, value in images.items():
        # Category by Category 0,1,..5
        if (debug):
            print("orb_descriptors for "+key)

        class_descriptors = []
        i = 0
        for img in value:
            # Loop over Dictionary items value by value M0 M1 M2
            # Every image in the category we are in
            keypoints, descriptors = orb.detectAndCompute(img, None)
            if (debug):
                print(str(i)+" has: " + str(len(keypoints)) + "Keypoints")
                i = i+1

            # Normalize descriptors
            if descriptors is None:
                print("No Key Points Detected")

            descriptor_list.extend(descriptors)
            class_descriptors.append(descriptors)

        # print("shape", np.shape(class_descriptors))
        # print("shape", np.shape(class_descriptors[0]))
        # print("shape", (class_descriptors[0]))

        # print(sift_vectors[key])
        orb_vectors[key]=class_descriptors  # [[keypts_img1*128],[keypts_img2*128],[keypts_img3*128]]
        # print(sift_vectors[key])
    return descriptor_list, orb_vectors




def image_feature_vectors(bag_of_words, centroids):
    """
    @param bag_of_words: dictionary {'0':[[img1(cluster*1)],[img1(cluster*1),[img2(cluster*1)]],..],'1':[],....'5':[]}
    @param centroids:  array of centroid of k means

    @return  a dictionary that holds the histograms for each images that are separated class by class. 
    ie. feature Vector(clusters*1) of every image
    @return classification: classification of Features
    """
    dict_feature = {}
    classification = np.array([])
    for key, value in bag_of_words.items():
        # print(key, np.shape(value))
        # Each Category
        category = []
        for img in value:
            # Each Image in Category
            ind, histogram = findClosestCentroids(X=img, centroids=centroids)
            category.append(histogram)  # Add this histogram to this Category
            classification=np.append(classification,key)
        dict_feature[key] = category


    #Dictionary rto the array
    all_feature_descriptors=[]

    for fd in dict_feature.values():
    # fd=fd.reshape(-1,8)
        for fv in fd:
            # print(fv)
            all_feature_descriptors.append(fv)

    all_feature_descriptors=np.squeeze(all_feature_descriptors)

    return all_feature_descriptors, classification