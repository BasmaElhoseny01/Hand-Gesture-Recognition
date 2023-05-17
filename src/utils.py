# Imports
import math
import cv2
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.pyplot import bar

from skimage.exposure import histogram
from skimage.feature import hog

from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier

from itertools import compress
import shutil
import os
import joblib
import sys

def test_import():
    print("Hello From utils")
def show_images(images, titles=None, save=False, path_save=""):
    """
    This function is used to show image(s) with titles by sending an array of images and an array of associated titles.
    @param images :array of images to be shown
    @param titles:titles corresponding to images

    @return None
    """
    #
    # images[0] will be drawn with the title titles[0] if exists
    # You aren't required to understand this function, use it as-is.
    n_ims = len(images)
    if titles is None:
        titles = ['(%d)' % i for i in range(1, n_ims + 1)]
    fig = plt.figure()
    n = 1
    for image, title in zip(images, titles):
        a = fig.add_subplot(1, n_ims, n)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
        n += 1
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_ims)
    if (save):
        plt.savefig(path_save)
    else:
        plt.show()
    return None


def read_image(path, color_space="RGB"):
    """
    Function to read image from a given path and returns image in a specific space

    @param path: relative path to the image
    @param color_space:space of in which we want the image

    @return image read in the passed space
    """
    image = cv2.imread(path)
    if (color_space == "RGB"):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

# Show Histogram of Gray Scale image
def showHist(img):
    """
    Function to how Histogram of Gray Scale image

    @param path: relative path to the image
    @param color_space:space of in which we want the image

    @return image read in the passed space
    """
    # An "interface" to matplotlib.axes.Axes.hist() method
    plt.figure()
    imgHist = histogram(img, nbins=256)
    bar(imgHist[1].astype(np.uint8), imgHist[0], width=0.8, align='center')


def read_images_test(path_data_folder, type="train"):
    """
    Read images for men and women
    """
    # Read Images in A dictionary
    images_men = images_Dictionary(path_data_folder+"men/"+type)
    # print(np.shape(images_men['0']))

    # Add Women Images
    # images_women = images_Dictionary(path_data_folder+"women/"+type)
    # print(images_women)
    # print(np.shape(images_women['0']))
    images = {'0': None, '1': None, '2': None,
              '3': None, '4': None, '5': None}

    # for i in range(0, 6):
    #     print(i)
    #     images[str(i)] = np.concatenate(
    #         (images_men[str(i)], images_women[str(i)]), axis=0)
    # Solution_01:Concatinate part by part, and delete each concatinated part
    images = images_men

    return images

def read_images_train(path_data_folder, debug=False):
    """
    Read images for men and women
    """
    # Read Images in A dictionary
    images_men = images_Dictionary(path_data_folder+"men/train/", debug=debug)
    # print(np.shape(images_men['0']))

    # Add Women Images
    images_women = images_Dictionary(
        path_data_folder+"women/train/", debug=debug)
    # print(images_women)
    # print(np.shape(images_women['0']))
    images = {'0': None, '1': None, '2': None,
              '3': None, '4': None, '5': None}

    for i in range(0, 6):
        print(i)
        images[str(i)] = np.concatenate(
            (images_men[str(i)], images_women[str(i)]), axis=0)
    # Solution_01:Concatenate part by part, and delete each concatenated part

    return images


def images_Dictionary(path_data_folder, debug=False):
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
    for filename in os.listdir(path_data_folder):
        # Each Subfolder

        path = path_data_folder + "/" + filename
        category_imgs = []
        for cat in os.listdir(path):
            # Every folder --> loop of all images in the folder of men
            img = cv2.imread(path + "/" + cat)  # YALAHWII
            # print(path + "/" + cat)
            if img is not None:
                if (debug):
                    show_images([img])
                category_imgs.append(img)
        if (images.get(filename) is None):
            images.update({filename: category_imgs})
        else:
            images[filename].append(category_imgs)

    return images


def kmeans_visual_words(k, descriptor_list, n_init=1):
    """
    Get Visual Words
    @param k:number of clusters
    @param descriptor_list: list  of all classes of all images
    @n_init:????

    @return visual_words: (*K) centroids of the k clusters   case SIFT (128*K) :)
    """
    kmeans = KMeans(n_clusters=k, n_init=n_init)
    kmeans.fit(descriptor_list)
    visual_words = kmeans.cluster_centers_
    return visual_words


def findClosestCentroids(X, centroids):
    """
    Returns the closest centroids in idx for a dataset X
    where each row is a single example. idx = m x 1 vector
    of centroid assignments (i.e. each entry in range [1..K])
    Args:
        X        : array(# training examples, n)
        centroids: array(K, n)
    Returns:
        idx      : array(# training examples, 1)
    """
    # Set K size.
    K = centroids.shape[0]
    count_clusters = np.zeros((K, 1))

    # Initialise idx.
    idx = np.zeros((X.shape[0], 1), dtype=np.int8)

    # Alternative partial vectorized solution.
    # Iterate over training examples.
    for i in range(X.shape[0]):
        distances = np.linalg.norm(X[i] - centroids, axis=1)
        # argmin returns the indices of the minimum values along an axis,
        # replacing the need for a for-loop and if statement.
        min_dst = np.argmin(distances)
        idx[i] = min_dst
        count_clusters[min_dst] += 1

    return idx, count_clusters


def draw_keypoints(img, keypoints, color=(255), radius=8, thickness=-1):
    # Convert Gray Scale to RGB ti be able to draw on it colors
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    for kp in keypoints:
        x, y = kp.pt
        # you can change the radius and the thickness
        cv2.circle(img, (int(x), int(y)), color=color,
                   radius=radius, thickness=thickness)

    return img

def getGraylevelCounts(image):
    """
        Documentation
    """
    histImage = histogram(image)  
    frequency = histImage[0]; bins = histImage[1]
    graylevels = np.zeros(256).astype(int)
    counts = np.zeros(256).astype(int)
    
    for i in range (0,256):
        graylevels[i] = i
    
    for i in range (0,frequency.shape[0]):
        counts[bins[i]] = frequency[i]

    return counts,graylevels

def getSegmentedImage(image, threshold):
    """
        Documentation
    """
    segmented_image = np.copy(image)
    segmented_image[segmented_image <= threshold] = 0
    segmented_image[segmented_image > threshold] = 255
    return segmented_image

def getThreshold(image):
    """
        Documentation
    """
    counter = 0
    image = image.astype(np.uint8)
    counts,bins = getGraylevelCounts(image)
    cumulativecount = np.cumsum(counts)
    t_old = 0

    threshold = round(np.sum(np.multiply(counts,bins)/cumulativecount[-1]))

    while(threshold != t_old):
        if(counter > 256):
            break
        counter +=1
        t_old = threshold
        low = list(range(0,t_old))
        high = list(range(t_old+1, 256))
        t_low = np.sum(np.multiply(counts[0:t_old], low))//cumulativecount[t_old-1]
        t_high = np.sum(np.multiply(counts[t_old+1:256],high))//(cumulativecount[-1]-cumulativecount[t_old+1])
        threshold = round((t_low + t_high)//2)
        # print("old threshold",str(t_old)) 
        # print("new threshold",str(threshold)) 
    return threshold

def gammaCorrection(src,gamma):
    invGamma=1/gamma
    table=[((i/255)**invGamma)*255 for i in range(256)]
    table=np.array(table,np.uint8)
    return cv2.LUT(src,table)
