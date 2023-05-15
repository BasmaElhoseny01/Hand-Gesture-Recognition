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
from sklearn.ensemble import RandomForestClassifier
from itertools import compress
import joblib
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

def read_images(path_data_folder, debug=False):
    """
    Read images for men and women
    """
    # Read Images in A dictionary
    images_men = images_Dictionary(path_data_folder+"men/train/", debug=debug)
    # print(np.shape(images_men['0']))

    # Add Women Images
    images_women = images_Dictionary(path_data_folder+"women/train/", debug=debug)
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
            img = cv2.imread(path + "/" + cat) #YALAHWII
            print(path + "/" + cat)
            if img is not None:
                if(debug):
                    show_images([img])
                category_imgs.append(img)
        if (images.get(filename) is None):
            images.update({filename: category_imgs})
        else:
            images[filename].append(category_imgs)

    return images