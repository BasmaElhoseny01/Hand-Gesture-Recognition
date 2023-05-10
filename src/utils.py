# Imports
import math
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.exposure import histogram
from matplotlib.pyplot import bar
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score

import shutil
import os


def show_images(images, titles=None,save=False,path_save=""):
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
    if(save):
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


def kmeans_visual_words(k, descriptor_list,n_init=1):
    """
    Get Visual Words
    @param k:number of clusters
    @param descriptor_list: list  of all classes of all images
    @n_init:????

    @return visual_words: (*K) centroids of the k clusters   case SIFT (128*K) :)
    """
    kmeans = KMeans(n_clusters = k, n_init=n_init)
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
    count_clusters=np.zeros((K,1))
    

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
        count_clusters[min_dst]+=1

    return idx,count_clusters


def draw_keypoints(img, keypoints, color=(255),radius=8,thickness=-1):
    # Convert Gray Scale to RGB ti be able to draw on it colors
    img=cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
    

    for kp in keypoints:
        x, y = kp.pt
        cv2.circle(img, (int(x), int(y)), color=color, radius=radius, thickness=thickness) # you can change the radius and the thickness

    return img

def showHist(img):
    # An "interface" to matplotlib.axes.Axes.hist() method
    plt.figure()
    imgHist = histogram(img, nbins=256)
    
    bar(imgHist[1].astype(np.uint8), imgHist[0], width=0.8, align='center')