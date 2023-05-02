# Imports
import math
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.exposure import histogram
from matplotlib.pyplot import bar

import os


def show_images(images, titles=None):
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
