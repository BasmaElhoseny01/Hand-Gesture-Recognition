import math
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.exposure import histogram
from matplotlib.pyplot import bar
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from skimage.feature import hog
from skimage.transform import resize
from sklearn import svm
import joblib
import shutil
import os

def read_images(path_data_folder):
    """
    Read images for men and women
    """
    # Read Images in A dictionary
    images_men = images_Dictionary(path_data_folder+"men/")
    # print(np.shape(images_men['0']))

    # Add Women Images
    images_women = images_Dictionary(path_data_folder+"women/")
    # print(images_women)
    # print(np.shape(images_women['0']))
    images = {'0': None, '1': None, '2': None,
              '3': None, '4': None, '5': None}

    for i in range(0, 6):
        print(i)
        images[str(i)] = np.concatenate(
            (images_men[str(i)], images_women[str(i)]), axis=0)
    # Solution_01:Concatinate part by part, and delete each concatinated part

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
    for filename in os.listdir(path_data_folder):
        # Each Subfolder

        path = path_data_folder + "/" + filename
        category_imgs = []
        for cat in os.listdir(path):
            # Every folder --> loop of all images in the folder of men
            img = cv2.imread(path + "/" + cat, cv2.IMREAD_GRAYSCALE)
            print(path + "/" + cat)
            if img is not None:
                img = resize(img, (64*4, 128*4))
                category_imgs.append(img)
        if (images.get(filename) is None):
            images.update({filename: category_imgs})
        else:
            images[filename].append(category_imgs)

    return images

def hog_features(images):
    X = []

    for img in images:
        img = resize(img, (64*4, 128*4))
        fd = hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
        X.append(fd)  
    return X