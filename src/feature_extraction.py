from utils import *


def extract_features(path_data_folder, clusters, debug=False, images=None, train=False, visual_words=None, orb_params=None):
    """
    clusters:Number of clusters for the Visual bag Kmeans clustering
    """

    # # Step(1)Read Images
    # images = read_images(path_data_folder)
    # if (debug):
    #     print("Read 0:", np.shape(images['0']))
    #     print("Read 1:", np.shape(images['1']))
    #     print("Read 2:", np.shape(images['2']))
    #     print("Read 3:", np.shape(images['3']))
    #     print("Read 4:", np.shape(images['4']))
    #     print("Read 5:", np.shape(images['5']))

    # Step(2) Descriptors
    # SIFT
    # descriptor_list, sift_vectors = sift_descriptors(images, debug=False)

    # ORB
    descriptor_list, orb_vectors = orb_descriptors(
        images, orb_params, debug=False)

    if debug:
        print('Description Done')
    # #Step(3) Visual Dictionary
    # #CHECK: To create visual dictionary, we only use train dataset.
    if (train):
        visual_words = kmeans_visual_words(
            clusters, descriptor_list)  # Centroids of the K clusters
        # Till here we  have Defined to have a feature vector of size = clusters
    if debug:
        print('Kmeans DONE')
    # # #Step(4) Feature Vector
    feature_vectors, classification = image_feature_vectors(
        orb_vectors, visual_words)
    # # Here We have Achieved Feature Vector size k :D for all images
    if debug:
        print('Feature vector DONE')
    # Conver Feature_vecrots from dictionary

    return feature_vectors, classification, visual_words


def read_images(path_data_folder, type="train"):
    """
    Read images for men and women
    """
    # Read Images in A dictionary
    images_men = images_Dictionary(path_data_folder+"men/"+type)
    # print(np.shape(images_men['0']))

    # Add Women Images
    images_women = images_Dictionary(path_data_folder+"women/"+type)
    # print(images_women)
    # print(np.shape(images_women['0']))
    images = {'0': None, '1': None, '2': None,
              '3': None, '4': None, '5': None}

    for i in range(0, 6):
        print(i)
        images[str(i)] = np.concatenate(
            (images_men[str(i)], images_women[str(i)]), axis=0)

    return images


def images_Dictionary(path_data_folder):
    """
    Folder Structure
    'class1'
        1.jpg
        .....
        50.jpg

    'class2'
        1.jpg
        .....
        80.jpg


    @param Data path
    @param images = {} to store in it
    """
    images = {
    }  # {'0':[[img1][img2]],'1':[[img1],[img2]...],'5':[[img1][img2]]}
    print(os.listdir(path_data_folder))
    for filename in os.listdir(path_data_folder):
        # Each Subfolder
        # if(filename==)

        path = path_data_folder + "/" + filename
        category_imgs = []
        for cat in os.listdir(path):
            # Every folder --> loop of all images in the folder of men
            img = cv2.imread(path + "/" + cat, cv2.IMREAD_GRAYSCALE)
            # print(path + "/" + cat)
            if img is not None:
                # img=cv2.resize(img,(1296,2304))
                img = cv2.resize(img, (np.shape(img)[1]//4,np.shape(img)[0]//4))
                category_imgs.append(img)
        if (images.get(filename) is None):
            images.update({filename: category_imgs})
        else:
            images[filename].append(category_imgs)

    return images


def sift_descriptors(images, debug=False):
    """
    Function to return all descriptors for all images according to the descriptor[Feature extractor used] 
    """

    # SIFT Feature Extractor
    sift = cv2.SIFT_create()

    # Dictionary {'0':[[keypts_of_img1*128],[keypts_of_img2*128],[keypts_of_img3*128]..]}
    sift_vectors = {'0': None, '1': None,
                    '2': None, '3': None, '4': None, '5': None}
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
        # [[keypts_img1*128],[keypts_img2*128],[keypts_img3*128]]
        sift_vectors[key] = class_descriptors
        # print(sift_vectors[key])
    return descriptor_list, sift_vectors


def orb_descriptors(images, params=None, debug=False):
    # initialize ORB detector with custom parameters
    if params is not None:
        orb = cv2.ORB_create(params[0], params[1], params[2])
        #                    nFeatures, scaleFactor, nLevels
    else:
        orb = cv2.ORB_create()
    # Dictionary {'0':[[keypts_of_img1*128],[keypts_of_img2*128],[keypts_of_img3*128]..]}
    orb_vectors = {'0': None, '1': None,
                   '2': None, '3': None, '4': None, '5': None}
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
                print("No Key Points Detected Dropped")

            else:
                descriptor_list.extend(descriptors)
                class_descriptors.append(descriptors)

        # print("shape", np.shape(class_descriptors))
        # print("shape", np.shape(class_descriptors[0]))
        # print("shape", (class_descriptors[0]))

        # print(sift_vectors[key])
        # [[keypts_img1*128],[keypts_img2*128],[keypts_img3*128]]
        orb_vectors[key] = class_descriptors
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
            classification = np.append(classification, key)
        dict_feature[key] = category

    # Dictionary rto the array
    all_feature_descriptors = []

    for fd in dict_feature.values():
        # fd=fd.reshape(-1,8)
        for fv in fd:
            # print(fv)
            all_feature_descriptors.append(fv)

    all_feature_descriptors = np.squeeze(all_feature_descriptors)

    return all_feature_descriptors, classification

# def extract_features(directory_path, num_of_clusters=3, debug=False):
#     all_feature_descriptors = []
#     all_expected_classes = []

#     for i in range(0, 6):
#         if(debug):
#             print(i)
#         result_dir_path = directory_path + 'men/' + str(i) + '/'
#         feature_descriptors, expexted_classes = descript_features(path_of_the_directory=result_dir_path, classification=i)
#         for fd in feature_descriptors:
#             all_feature_descriptors.append(fd)
#         for ec in expexted_classes:
#             all_expected_classes.append(ec)
#     for i in range(0, 6):
#         if(debug):
#             print(i)
#         result_dir_path = directory_path + 'women/' + str(i) + '/'
#         feature_descriptors, expexted_classes = descript_features(path_of_the_directory=result_dir_path, classification=i)
#         for fd in feature_descriptors:
#             all_feature_descriptors.append(fd)
#         for ec in expexted_classes:
#             all_expected_classes.append(ec)
#     feature_vector = cluster_descriptors(all_feature_descriptors, num_of_clusters)

#     return feature_vector, all_expected_classes

# def descript_features(path_of_the_directory, classification, debug=False):
#     feature_descriptor = []     # size: i * n * 128 , i is the number of images, n is the number of key points
#     expected_class = []         # List of len = i, i is the number of images

#     for filename in os.listdir(path_of_the_directory):
#         img = cv2.imread(path_of_the_directory+str(filename), cv2.IMREAD_GRAYSCALE)
#         corners_descriptors=Harris(img, debug)
#         feature_descriptor.append(corners_descriptors)
#         expected_class.append(classification)

#     return feature_descriptor, expected_class

# # ORB
# # SIFT
# # SURF

# def cluster_descriptors(feature_descriptor, num_of_clusters=3):
#     #Get the descriptors of the found key points in the images
#     descriptors = [j for k in feature_descriptor if k is not None for j in k]   #Cleaner version
#     num_of_images = len(feature_descriptor)
#     kmeans = KMeans(n_clusters=num_of_clusters).fit(descriptors)
#     final_fv = np.zeros((num_of_images, num_of_clusters))
#     k_indx = 0
#     for img_indx in range(len(feature_descriptor)):
#         if feature_descriptor[img_indx] is not None:
#             for p in range(len(feature_descriptor[img_indx])):
#                 final_fv[img_indx][kmeans.labels_[k_indx]] += 1
#                 k_indx += 1
#     return final_fv

# def Harris(img, debug=False):
#     '''
#     img:"grey"
#     '''
#     block_size = 2
#     aperture_size = 3
#     k = 0.04
#     harris_img = cv2.cornerHarris(img, block_size, aperture_size, k)

#     harris_img = cv2.normalize(harris_img, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
#     # Create SIFT descriptor
#     sift = cv2.SIFT_create()

#     # Detect and compute SIFT descriptors for Harris corners
#     # Normalize
#     harris_img = cv2.normalize(
#         harris_img, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
#     keypoints, descriptors = sift.detectAndCompute(harris_img, None)

#     if (debug):
#         output_image = cv2.drawKeypoints(img, keypoints, 0, (255, 0, 0),
#                                          flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
#         show_images([output_image], ['Key points'])

#     return descriptors

# CORNER DETECTION


# def detectCorners(filename):
#     # read the image
#     img = cv2.imread(filename)
#     # plt.imshow(img), plt.show()

#     # img = cv2.dilate(img, (50, 20), img)
#     # plt.imshow(img), plt.show()

#     # convert image to gray scale image
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#     # detect corners with the goodFeaturesToTrack function.
#     corners = cv2.goodFeaturesToTrack(gray, 0, 0.8, 10)
#     corners = np.int0(corners)
#     # print(corners[0][0])
#     corners_detected = []
#     # we iterate through each corner,
#     # making a circle at each point that we think is a corner.
#     for corner in corners:
#         x, y = corner.ravel()
#         # print(x, y)
#         corners_detected.append([x, y])
#         centre = (x, y)
#         color = (255, 0, 0)
#         radius = 50
#         thickness = 10

#         img = cv2.circle(img, centre, radius, color, thickness)

#     corners_detected = np.asarray(corners_detected)

#     # print(corners_detected)

#     # plt.imshow(img), plt.show()

#     return corners_detected


def OCR(features_vector,img,name):
    '''img:Binary Assume images are same size :D'''
    sum_cols = np.sum(img,axis=0)
    if(np.shape(sum_cols)[0]!=np.shape(features_vector)[1]):
        print("Mismatch in Size",name," has ",np.shape(sum_cols)[0] ,'while before has ',np.shape(features_vector)[1])
        return features_vector
    
    features_vector_train=np.vstack([features_vector_train,sum_cols])
    return features_vector