import cv2
import numpy as np
import matplotlib.pyplot as plt
from itertools import compress
import cv2
import numpy as np
import utils
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

def gammaCorrection(src,gamma):
    invGamma=1/gamma
    table=[((i/255)**invGamma)*255 for i in range(256)]
    table=np.array(table,np.uint8)
    return cv2.LUT(src,table)


def shadow_remove(img):
    '''
    img:bgr img
    '''
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)#Convert it to Gray

    rgb_planes = cv2.split(img)
    result_norm_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        # print(diff_img)
        norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        result_norm_planes.append(norm_img)
    shadow_removed = cv2.merge(result_norm_planes)
    return shadow_removed

def preprocessing_new_2(img,name="",debug=False):
    '''
    1.Dec size to 1/4
    2.Remove shadow
    3.Gamma Correction for the removal of shadow
    4.Canny
    5.Adding Padding
    6.Draw line to right + Region Filling
    7.Draw line to left + Region Filling
    8.Flip orientation
    9.Find Contours

    '''
    '''
    img:rgb
    @return binary img
    '''

    # Resize -------------------------------------------------------------------------------------------------------
    img = cv2.resize(img, (np.shape(img)[1]//4,np.shape(img)[0]//4))  
    # img = cv2.resize(img, (128*4,64*4))
    # show_images([img],['img'])    
    #--------------------------------------------------------------------------------------------------------
    #Shadow removal
    shadow_removed = shadow_remove(img)

    #----------------------------------------------------------------------------------------------------------------
    #Gamma Correction
    # shadow_removed_gamma=gammaCorrection(shadow_removed,0.4)
    shadow_removed_gamma=shadow_removed
    shadow_removed_gamma=cv2.cvtColor(shadow_removed_gamma,cv2.COLOR_RGB2GRAY)#Convert it to Gray


    #-------------------------------------------------------------------------------------------------------------
    #Canny Edge
    # Setting parameter values
    t_lower = 50  # Lower Threshold
    t_upper = 120  # Upper threshold
    
    # Applying the Canny Edge filter
    edge = cv2.Canny(shadow_removed_gamma, t_lower, t_upper)
    # Erosion
    kernel = np.ones((5, 5), np.uint8)
    # edge = cv2.morphologyEx(edge, cv2.MORPH_DILATE, kernel, iterations=5) #erode
    edge = cv2.morphologyEx(edge, cv2.MORPH_CLOSE, kernel, iterations=5) #erode

    #--------------------------------------------------------------------------------------------------------------------
    # Add Padding
    edge=np.pad(edge,pad_width=50,mode='constant',constant_values=0)


    # Draw Line to the right --------------------------------------------------------------------------------------
    edge[20:-20,-120:-100]=255

    # Region Filling
    img_fill=edge.copy()
    h,w=edge.shape[:2]
    mask=np.zeros((h+2,w+2),np.uint8)
    cv2.floodFill(img_fill,mask,(0,0),255)#img_fill marks regions filled -> not so that we can see it bec they are black
    #mask 0,1 while bit wise not get 255,254
    region_filling_right=cv2.bitwise_not(mask*255)//255
    #Remove Line
    region_filling_right[20:-20,-120:]=0
    edge[20:-20,-120:-100]=0


    edge=region_filling_right

    # Draw Line to the left --------------------------------------------------------------------------------------
    edge[20:-20,100:120]=255

    # Region Filling
    img_fill=edge.copy()
    h,w=edge.shape[:2]
    mask=np.zeros((h+2,w+2),np.uint8)
    cv2.floodFill(img_fill,mask,(0,0),255)#img_fill marks regions filled -> not so that we can see it bec they are black
    #mask 0,1 while bit wise not get 255,254
    region_filling_left=cv2.bitwise_not(mask*255)//255

    #Remove Line
    region_filling_left[20:-20,0:120]=0
    edge[20:-20,100:120]=0


    #Erode Dilate
    region_filling=region_filling_left

    kernel = np.ones((3, 3), np.uint8)
    region_filling = cv2.morphologyEx(region_filling, cv2.MORPH_DILATE, kernel, iterations=2) #erode


    # Erosion for smoothing
    kernel = np.ones((2, 2), np.uint8)
    # edge = cv2.morphologyEx(region_filling, cv2.MORPH_DILATE, kernel, iterations=5) #erode
    region_filling = cv2.morphologyEx(region_filling, cv2.MORPH_ERODE, kernel, iterations=5) #erode


    #-----------------------------------------------------------------------------------------------------------
    #Flip 
    _,_,_,img_flip=flip_horizontal(region_filling,debug)


    if(debug):
        utils.show_images([cv2.cvtColor(img,cv2.COLOR_BGR2RGB),shadow_removed,shadow_removed_gamma,edge,region_filling],['Original','shadow_removed','shadow_removed_gamma,edge','region_filling'])
        utils.show_images([img_flip],['img_flip'])
    

    #-----------------------------------------------------------------------------------------------------------------
    # Find Contours
    contours, hierarchy = cv2.findContours(
            img_flip, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    if(len(contours)==0):
        print("No Contours found",name)
        return None

    # Draw for debug
    if (debug):
        img_contours = np.copy(img)
        cv2.drawContours(img_contours, contours, -1, (0, 255, 0), 3)

    # Get Largest Contour
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
    largest_contour = sorted_contours[0]

    # Binary_img_contours[Result]
    hand_contour = np.zeros((np.shape(img_flip)[0], np.shape(img_flip)[1], 1))
    cv2.drawContours(hand_contour, largest_contour, -1, 255, 10)


    if(debug):
        print("Contours",np.shape(contours))
        utils.show_images([img_contours,hand_contour])


    return img_flip*255,hand_contour
    
        
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