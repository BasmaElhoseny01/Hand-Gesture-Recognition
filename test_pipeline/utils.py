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
            img = cv2.imread(path + "/" + cat)
            if img is not None:
                category_imgs.append(img)
        if (images.get(filename) is None):
            images.update({filename: category_imgs})
        else:
            images[filename].append(category_imgs)

    return images

def performance_analysis(result,expected):
    # Accuracy is the percentage of data that are correctly classified, which ranges from 0 to 1
    accuracy=accuracy_score(result, expected)
    
    # Precision is your go-to evaluation metric when dealing with imbalanced data.

    # Recall
    return accuracy