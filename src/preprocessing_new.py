import cv2
import numpy as np
import matplotlib.pyplot as plt
from itertools import compress

from utils import show_images

def preprocessing_new_1(img,debug=False):
    '''
    1.Dec size to 1/4
    2.Equalize S
    3.Remove BackGround
    4.Flip orientation
    5.Find Contours

    Commented
    -Equalize Y in YUV 
    -Region Filling
    -Shift Center to right
    -Draw Circle
    '''
    # Decrease Size to Quarter
    img = cv2.resize(img, (np.shape(img)[1]//4,np.shape(img)[0]//4))
    
    #------------------------------------------------------------------------------------------------------------------------------
    # Equalize S
    img_hsv=cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    #Equalize S
    # # img_HSV_eq=np.copy(img_hsv)
    # # img_hsv[:,:,0]=cv2.equalizeHist(img_hsv[:,:,0])
    img_hsv[:,:,1]=cv2.equalizeHist(img_hsv[:,:,1])
    # # img_hsv[:,:,2]=cv2.equalizeHist(img_hsv[:,:,2])


    # img_YUV=cv2.cvtColor(img,cv2.COLOR_BGR2YUV)
    # Y,U,V=cv2.split(img_YUV)
    # Y_Eq=cv2.equalizeHist(Y)
    # img_eq=cv2.merge([Y_Eq,U,V])


    #-----------------------------------------------------------------------------------------------------------------------------
    #Remove BG

    # Defining HSV Thresholds
    lower_threshold = np.array([0, 48, 80], dtype=np.uint8)
    upper_threshold = np.array([20, 255, 255], dtype=np.uint8)

    # Single Channel mask,denoting presence of colours in the about threshold
    skinMask_b = cv2.inRange(img_hsv, lower_threshold, upper_threshold)//255

    # Cleaning up mask using Gaussian Filter
    skinMask = cv2.GaussianBlur(skinMask_b, (3, 3), 0)


    
    # Erosion 
    kernel = np.ones((3, 3), np.uint8)
    skinMask = cv2.morphologyEx(skinMask, cv2.MORPH_ERODE, kernel, iterations=4) #erode

    if(debug):
        # Extracting skin from the threshold mask  [Debug]
        skin = cv2.bitwise_and(img_hsv, img_hsv, mask=skinMask)
        img_eq=cv2.cvtColor(skin,cv2.COLOR_HSV2RGB) #Extracted Colored Hand


    #----------------------------------------------------------------------------------------------------------------------------------
    # # Region Filling 
    # img_fill=skinMask.copy()
    # h,w=skinMask.shape[:2]
    # mask=np.zeros((h+2,w+2),np.uint8)

    # # cv2.floodFill(img_fill,mask,(0,0),255)#img_fill marks regions filled -> not so that we can see it bec they are black
    # # img_fill=cv2.bitwise_not(img_fill)

    # region_filling=cv2.bitwise_not(mask)


    #Flip 
    _,_,_,img_flip=flip_horizontal(skinMask,debug)


    #Shift image to the right
    # img_flip=img_flip[:,0:hand_center_x+1]

    #Draw circle
    # print('hand center',hand_center_x)
    # image_finger = cv2.circle(img_flip, (hand_center_x,np.shape(img_flip)[0]//2), 400,0, -1)


    # Find Contours #############################################################################################33
    contours, hierarchy = cv2.findContours(
            img_flip, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    


    # Draw for debug
    if (debug):
        img_contours = np.copy(img_flip)
        cv2.drawContours(img_contours, contours, -1, (0, 255, 0), 3)

    # Get Largest Contour
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
    largest_contour = sorted_contours[0]

    # Binary_img_contours[Result]
    hand_contour = np.zeros((np.shape(img_flip)[0], np.shape(img_flip)[1], 1))
    cv2.drawContours(hand_contour, largest_contour, -1, 255, 10)

    return hand_contour


def flip_horizontal(img,debug=False):
    '''
    img:Binary image
    '''

    #Compute OCR
    OCR = np.sum(img,axis=0)


    #Get Max index
    res=list(compress(range(len(OCR==np.max(OCR))),OCR==np.max(OCR)))
    max_x_ind=res[0]

    #Get Min index
    result = min(enumerate(OCR), key=lambda x: x[1] if x[1] > 20 else float('inf'))  #CHECK: Handle inf case
    # print("Position : {}, Value : {}".format(*result))
    min_x_ind=result[0]


    if(min_x_ind>max_x_ind):
        img=cv2.flip(img,1) #1=horizontally
        max_x_ind=np.shape(img)[1]-max_x_ind
        min_x_ind=np.shape(img)[1]-min_x_ind 
      


    if(debug):
        print("Image Size",np.shape(img))
        print("OCR shape",np.shape(OCR))
        print("Max x is at",max_x_ind)
        print("Min x is at",min_x_ind)

        # x-coordinates of left sides of bars 
        left = range(0,np.shape(OCR)[0])
        print("X Range",left)
        
        # heights of bars
        height = OCR
        print("Heights",height)

        # # labels for bars  NOT WORKING
        # # tick_label = range(0,1152)
        # tick_label=np.arange(0, np.shape(sum_cols)[0], 1.0)
        
        # plotting a bar chart
        plt.bar(left, height,
                width = 0.1, color = ['red', 'green'])
        
        # naming the x-axis
        plt.xlabel('x - axis')
        # naming the y-axis
        plt.ylabel('y - axis')
        # plot title
        plt.title('My bar chart!')
        
        # function to show the plot
        plt.show()


        show_images([img],['Flipped'])


    return OCR,max_x_ind,min_x_ind,img