from utils import *


# def preprocessing(img, name="", debug=False, gamma=False, close=False):
#     '''
#     @parm img:BGR Scale img
#     '''
#     if (gamma):
#         img = gamma_trans(img, 4)  # WORKING

#     #Skin Mask
#     skin = skin_masks(img, name, debug)

#     if (close):
#         kernel = np.ones((5, 5), np.uint8)
#         skin = cv2.morphologyEx(skin, cv2.MORPH_CLOSE, kernel, iterations=10)
#     # Find Contours
#     contours, hierarchy = cv2.findContours(
#         skin, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

#     # Draw for debug
#     if (debug):
#         img_contours = np.copy(img)
#         cv2.drawContours(img_contours, contours, -1, (0, 255, 0), 3)

#     # Get Largest Contour
#     sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
#     largest_contour = sorted_contours[0]
#     if (debug):
#         largest_contour_img = np.copy(img)
#         cv2.drawContours(largest_contour_img,
#                          largest_contour, -1, (255, 0, 0), 10)

#     # Binary_img_contours[Result]
#     hand_contour = np.zeros((np.shape(img)[0], np.shape(img)[1], 1))
#     cv2.drawContours(hand_contour, largest_contour, -1, 255, 10)

#     if (debug):
#         show_images([cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
#                     skin, img_contours, largest_contour_img, hand_contour],
#                     ['RGB'+name, 'Mask(3)YCrCb', 'Contours', 'Largest Contour', 'hand_contour'])

#     return hand_contour


# def remove_shadow(img):
#     """
#     Removes shadow form bgr image based on LAB color space

#     @param img:BGR image

#     @return BGR Image without shadow
#     """
#     # Convert to Gray Scale

#     img_Gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#     shadow = img_Gray < 150

#     show_images([img, img_Gray, shadow], ['BGR', 'Gray', 'Shadow 150'])
#     return None


# def skin_masks(img_bgr, name="", debug=False):
#     '''
#     Gets different masks for the skin color on the image
#     @parm img_bgr:BGR Scale img

#     @return: Binary Image of Skin Area
#     '''
#     # HSV Space
#     # img_HSV = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
#     # HSV_mask = cv2.inRange(img_HSV, (0, 15, 0), (17, 170, 255))

#     # HSV_mask = cv2.morphologyEx(HSV_mask, cv2.MORPH_OPEN,
#     #                             kernel, iterations=3)

#     # YCrCb Space
#     img_YCrCb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)

#     # min_YCrCb = np.array([0, 133, 77], np.uint8)
#     # max_YCrCb = np.array([235, 173, 127], np.uint8)
#     # imageYCrCb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCR_CB)
#     # skinRegionYCrCb = cv2.inRange(imageYCrCb, min_YCrCb, max_YCrCb)
#     # skinRegionYCrCb = cv2.morphologyEx(skinRegionYCrCb, cv2.MORPH_OPEN,
#     #                                    kernel, iterations=3)

#     # YCrCb Space 2
#     YCrCb_mask = cv2.inRange(img_YCrCb, (0, 135, 85), (255, 180, 135))
#     kernel = np.ones((3, 3), np.uint8)
#     YCrCb_mask = cv2.morphologyEx(YCrCb_mask, cv2.MORPH_OPEN,
#                                   kernel, iterations=3)

#     # show_images([cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), HSV_mask, skinRegionYCrCb,
#     #             YCrCb_mask, ], ['RGB'+name, 'Mask(1) HSV', 'Mask(2)YCrCb', 'Mask(3)YCrCb'])
#     # if(debug):
#     # show_images([cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB),
#     #             YCrCb_mask, ], ['RGB'+name, 'Mask(3)YCrCb'])

#     return YCrCb_mask


# def ditch_specular(img_bgr):
#     # Convert the image to grayscale
#     gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
#     # Apply a median blur to reduce noise
#     blur = cv2.medianBlur(gray, 3)
#     # Apply a threshold to obtain the specular mask
#     # _, mask = cv2.threshold(blur, 240, 255, cv2.THRESH_BINARY_INV)
#     # Perform morphological operations to remove small objects and fill holes
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 10))
#     mask = cv2.morphologyEx(blur, cv2.MORPH_GRADIENT, kernel)
#     # Apply the mask to the original image to remove the specular component
#     result = cv2.bitwise_and(img_bgr, img_bgr, mask=mask)
#     return result


# def clahe(img_bgr):
#     grayscale = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
#     # Apply contrast limiting adaptive histogram equalization
#     clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(5, 50))
#     cl1 = clahe.apply(grayscale)
#     # Apply a threshold to obtain the specular mask
#     _, mask = cv2.threshold(cl1, 240, 255, cv2.THRESH_BINARY_INV)
#     # Perform morphological operations to remove small objects and fill holes
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 15))
#     mask = cv2.morphologyEx(cl1, cv2.MORPH_OPEN, kernel)

#     # Apply the mask to the original image to remove the specular component
#     result = cv2.bitwise_and(img_bgr, img_bgr, mask=mask)
#     return result


# def gamma_trans(img, gamma):
#     gamma_table = [np.power(x/255.0, gamma)*255.0 for x in range(256)]
#     gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
#     return cv2.LUT(img, gamma_table)


# def detect_exposure(img):
#     img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     hist, bins = np.histogram(img_gray, bins=256, range=(0, 255))
#     print(hist)
# # -----------------------------------------------------------------------------------------------------------------------------------------------------
# def preprocessing2(img, name="", debug=False):
#     '''
#     Uses new skin_mask_range()
#     @parm img:BGR Scale img
#     '''

#     # Get Skin
#     skinMask=extract_skin(img,name=name,debug=debug)


#     # Apply Closing =(Erode+Dilate) to remove 
#     kernel = np.ones((3, 3), np.uint8)
#     skinMask = cv2.morphologyEx(skinMask, cv2.MORPH_ERODE, kernel, iterations=2) #erode
#     skinMask = cv2.morphologyEx(skinMask, cv2.MORPH_DILATE, kernel, iterations=2) #Dilate Back  

#     # Find Contours
#     contours, hierarchy = cv2.findContours(
#         skinMask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

#     # Draw for debug
#     if (debug):
#         img_contours = np.copy(img)
#         cv2.drawContours(img_contours, contours, -1, (0, 255, 0), 3)

#     # Get Largest Contour
#     sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
#     largest_contour = sorted_contours[0]
#     if (debug):
#         largest_contour_img = np.copy(img)
#         cv2.drawContours(largest_contour_img,
#                          largest_contour, -1, (255, 0, 0), 10)

#     # Binary_img_contours[Result]
#     hand_contour = np.zeros((np.shape(img)[0], np.shape(img)[1], 1))
#     cv2.drawContours(hand_contour, largest_contour, -1, 255, 10)

#     if (debug):
#         show_images([cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
#                     skinMask, img_contours, largest_contour_img, hand_contour],
#                     ['RGB'+name, 'Mask(3)YCrCb', 'Contours', 'Largest Contour', 'hand_contour'])

#     return hand_contour

# def extract_skin(image,name="",debug=False):  #REPEATED
    
#     # Converting from BGR Colors Space to HSV
#     img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

#     # Defining HSV Threadholds
#     lower_threshold = np.array([0, 48, 80], dtype=np.uint8)
#     upper_threshold = np.array([20, 255, 255], dtype=np.uint8)

#     # Single Channel mask,denoting presence of colours in the about threshold
#     skinMask_b = cv2.inRange(img, lower_threshold, upper_threshold)

#     # Cleaning up mask using Gaussian Filter
#     skinMask = cv2.GaussianBlur(skinMask_b, (3, 3), 0)

#     if(debug):
#         # Extracting skin from the threshold mask
#         skin = cv2.bitwise_and(img, img, mask=skinMask)

#         # Return the Skin image
#         skin=cv2.cvtColor(skin, cv2.COLOR_HSV2RGB)
#         show_images([ cv2.cvtColor(image, cv2.COLOR_BGR2RGB),skinMask_b,skinMask,skin],['2_men (11).JPG','skinMask_b','skinMask','skin'])


#     return  skinMask

def RGB_Mask(img):
    '''
    Get RGB Mask of the Skin
    Ref:https://medium.com/swlh/face-detection-using-skin-tone-threshold-rgb-ycrcb-python-implementation-2d4f62d376f1


    img:BGR
    '''
    img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    R=img[:,:,0]
    G=img[:,:,1]
    B=img[:,:,2]

    BGR_Max=np.maximum.reduce([R,G,B])
    BGR_Min=np.minimum.reduce([R,G,B])

    Rule_1=np.logical_and.reduce([R>95,G>40,B>20,(BGR_Max-BGR_Min)>15,abs(R-G)>15,R>G,R>B])
    Rule_2=np.logical_and.reduce([R>220,G>210,B>170,abs(R-G)<=15,R>B,G>B])

    RGB_Rule=np.bitwise_or(Rule_1,Rule_2)
    RGB_Rule=RGB_Rule*255

   

    return RGB_Rule


def flip_orientation(img):
    '''
    Adjust Orientation of the hand Horizontally
    Still need to modify it vertically ***


    img:Binary image
    '''

    # Flip To make Same Orientation of the hand [Horizontally]
    sum_cols = np.sum(img,axis=0)

    OCR=sum_cols
    res=list(compress(range(len(sum_cols==np.max(OCR))),sum_cols==np.max(OCR)))

    if(res[0]<(np.shape(img)[1]/2)):
        img=cv2.flip(img,1) #1=horizontally

    return OCR,img



def preprocessing_OCR(img):
    '''
    img:BGR
    '''
    #Resize
    img = cv2.resize(img, (128*4,64*4))#width,height


    #Get Mask
    hand_mask=RGB_Mask(img)

    #Flip
    OCR,hand_mask=flip_orientation(hand_mask)
    return OCR,hand_mask


# ##########################################################################################################
def Get_Hand(img,debug=False):
    '''
    img:BGR
    '''
    # RGB_Rule
    hand_mask=RGB_Mask(img)

    #Flip
    _,hand_mask=flip_orientation(hand_mask)

    print(np.shape(hand_mask))
    print(np.max(hand_mask))
    print(np.min(hand_mask))
    print(len(hand_mask.shape))
    print(type(hand_mask))


    #Get Hand Contour
    contours, hierarchy = cv2.findContours(
        np.array(hand_mask), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    # Draw for debug
    if (debug):
        img_contours = np.copy(img)
        cv2.drawContours(img_contours, contours, -1, (0, 255, 0), 3)

    # Get Largest Contour
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
    largest_contour = sorted_contours[0]


    # if (debug):
    #     largest_contour_img = np.copy(img)
    #     cv2.drawContours(largest_contour_img,
    #                      largest_contour, -1, (255, 0, 0), 10)

    # Binary_img_contours[Result]
    hand_contour = np.zeros((np.shape(img)[0], np.shape(img)[1], 1))
    cv2.drawContours(hand_contour, largest_contour, -1, 255, 10)

    # if (debug):
    #     show_images([cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
    #                 skin, img_contours, largest_contour_img, hand_contour],
    #                 ['RGB'+name, 'Mask(3)YCrCb', 'Contours', 'Largest Contour', 'hand_contour'])

    return hand_contour


# def preprocessing_hagrass_eq_s(img,debug=False):
#     #Decrease Side of image ####################################################################################################################
#     img = cv2.resize(img, (np.shape(img)[1]//4,np.shape(img)[0]//4))

#     #Equalize S###############################################################################################################
#     img_hsv=cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

#     #Equalize S
#     # # img_HSV_eq=np.copy(img_hsv)
#     # # img_hsv[:,:,0]=cv2.equalizeHist(img_hsv[:,:,0])
#     img_hsv[:,:,1]=cv2.equalizeHist(img_hsv[:,:,1])
#     # # img_hsv[:,:,2]=cv2.equalizeHist(img_hsv[:,:,2])

#     # img_YUV=cv2.cvtColor(img,cv2.COLOR_BGR2YUV)
#     # Y,U,V=cv2.split(img_YUV)
#     # Y_Eq=cv2.equalizeHist(Y)

#     # img_eq=cv2.merge([Y_Eq,U,V])


#     # Remove  ############################################################################################################################

#     # Defining HSV Threadholds
#     lower_threshold = np.array([0, 48, 80], dtype=np.uint8)
#     upper_threshold = np.array([20, 255, 255], dtype=np.uint8)

#     # Single Channel mask,denoting presence of colours in the about threshold
#     skinMask_b = cv2.inRange(img_hsv, lower_threshold, upper_threshold)

#     # Cleaning up mask using Gaussian Filter
#     skinMask = cv2.GaussianBlur(skinMask_b, (3, 3), 0)
#     # show_images([skinMask])
#     # 

#     # Extracting skin from the threshold mask
#     skin = cv2.bitwise_and(img_hsv, img_hsv, mask=skinMask)

#     img_eq=cv2.cvtColor(skin,cv2.COLOR_HSV2RGB)
#     # img_eq=skinMask


#     # Ersion #########################################################################################################################
#     kernel = np.ones((3, 3), np.uint8)
#     skinMask = cv2.morphologyEx(skinMask, cv2.MORPH_ERODE, kernel, iterations=4) #erode


#     # # Region Filling ###########################################################################################################
#     # img_fill=skinMask.copy()
#     # h,w=skinMask.shape[:2]
#     # mask=np.zeros((h+2,w+2),np.uint8)

#     # # cv2.floodFill(img_fill,mask,(0,0),255)#img_fill marks regions filled -> not so that we can see it bec they are black
#     # # img_fill=cv2.bitwise_not(img_fill)

#     # region_filling=cv2.bitwise_not(mask)


#     #Flip ######################################################################################################################
#     # _,hand_center_x,img_flip=flip_orientation(region_filling)
#     _,hand_center_x,img_flip=flip_orientation2(skinMask)


#     #Shift image to the right
#     # img_flip=img_flip[:,0:hand_center_x+1]

#     # print('hand center',hand_center_x)
#     # image_finger = cv2.circle(img_flip, (hand_center_x,np.shape(img_flip)[0]//2), 400,0, -1)


#     # Find Contours #############################################################################################33
#     contours, hierarchy = cv2.findContours(
#             img_flip, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    


#     # Draw for debug
#     if (debug):
#         img_contours = np.copy(img_flip)
#         cv2.drawContours(img_contours, contours, -1, (0, 255, 0), 3)

#     # Get Largest Contour
#     sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
#     largest_contour = sorted_contours[0]

#     # Binary_img_contours[Result]
#     hand_contour = np.zeros((np.shape(img_flip)[0], np.shape(img_flip)[1], 1))
#     cv2.drawContours(hand_contour, largest_contour, -1, 255, 10)

#     return hand_contour

# def flip_orientation2(img):
#     '''
#     Adjust Orientation of the hand Horizontally
#     Still need to modify it vertically ***


#     img:Binary image
#     '''

#     # Flip To make Same Orientation of the hand [Horizontally]
#     sum_cols = np.sum(img,axis=0)

#     # # print(np.shape(sum_cols))
#     # # print(np.shape(sum_cols)[0]%4)
#     # remain=np.zeros((np.shape(sum_cols)[0]%2,),np.uint8)
#     # # print(np.shape(remain))

#     # sum_cols=np.concatenate((sum_cols, remain), axis=0)
#     # print(np.shape(sum_cols))

#     # # print(np.shape(sum_cols.reshape(4,-1)))
#     # sum_cols=sum_cols.reshape(2,-1)
#     # sum_cols = np.sum(sum_cols,axis=0)
#     # sum_cols=sum_cols.reshape(-1,1)
#     # print(np.shape(sum_cols))    


# # 
#     OCR=sum_cols
#     res=list(compress(range(len(sum_cols==np.max(OCR))),sum_cols==np.max(OCR)))

#     if(res[0]<(np.shape(img)[1]/2)):
#         img=cv2.flip(img,1) #1=horizontally
#         res[0]=np.shape(img)[1]-res[0]

#     #res is x with max sum
#     return OCR,res[0],img

def Cut_pre(img):
  img = cv2.resize(img, (np.shape(img)[1]//4,np.shape(img)[0]//4))
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

  # Defining HSV Threadholds
  lower_threshold = np.array([0, 48, 80], dtype=np.uint8)
  upper_threshold = np.array([20, 255, 255], dtype=np.uint8)

  # Single Channel mask,denoting presence of colours in the about threshold
  skinMask_b = cv2.inRange(img_hsv, lower_threshold, upper_threshold)

  # Cleaning up mask using Gaussian Filter
  skinMask = cv2.GaussianBlur(skinMask_b, (3, 3), 0)
  # show_images([skinMask])
# 

  # Extracting skin from the threshold mask
  skin = cv2.bitwise_and(img_hsv, img_hsv, mask=skinMask)

  img_eq=cv2.cvtColor(skin,cv2.COLOR_HSV2RGB)
  # img_eq=skinMask


  # Ersion
  kernel = np.ones((3, 3), np.uint8)
  skinMask = cv2.morphologyEx(skinMask, cv2.MORPH_ERODE, kernel, iterations=4) #erode


  # # Region Filling
  # img_fill=skinMask.copy()
  # h,w=skinMask.shape[:2]
  # mask=np.zeros((h+2,w+2),np.uint8)

  # # cv2.floodFill(img_fill,mask,(0,0),255)#img_fill marks regions filled -> not so that we can see it bec they are black
  # # img_fill=cv2.bitwise_not(img_fill)

  # region_filling=cv2.bitwise_not(mask)


  #Flip
  # _,hand_center_x,img_flip=flip_orientation(region_filling)
  OCR,hand_center_x,img_flip=flip_orientation3(skinMask)


  #Shift image to the right
  img_flip=img_flip[:,0:hand_center_x+1]

  # print('hand center',hand_center_x)
  image_finger = cv2.circle(img_flip, (hand_center_x,np.shape(img_flip)[0]//2), 400,0, -1)


#   # Find Contours
#   contours, hierarchy = cv2.findContours(
#           img_flip, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
  

#   debug=True

#   # Draw for debug
#   if (debug):
#       img_contours = np.copy(img_flip)
#       cv2.drawContours(img_contours, contours, -1, (0, 255, 0), 3)

#   # Get Largest Contour
#   sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
#   largest_contour = sorted_contours[0]

#   # Binary_img_contours[Result]
#   hand_contour = np.zeros((np.shape(img_flip)[0], np.shape(img_flip)[1], 1))
#   cv2.drawContours(hand_contour, largest_contour, -1, 255, 10)

#   #     if (debug):
#   #         show_images([cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
#   #                     skin, img_contours, largest_contour_img, hand_contour],
#   #                     ['RGB'+name, 'Mask(3)YCrCb', 'Contours', 'Largest Contour', 'hand_contour'])



#   show_images([cv2.cvtColor(img,cv2.COLOR_BGR2RGB),skinMask,img_eq,img_flip,img_contours,hand_contour,image_finger])

  # show_images([cv2.cvtColor(img,cv2.COLOR_BGR2RGB),skinMask,img_eq,region_filling,img_flip,img_contours,hand_contour])

  # # show_images([Y,Y_Eq],['Y_channel before Eq','Y After Eq'])
  # # show_images([img])



  # Kmeans Clustering
  # clusters=3
  # clt = KMeans(n_clusters = clusters,init='k-means++')

  # reshape the image to be a list of pixels
  # image = img_eq.reshape((img_eq.shape[0] * img_eq.shape[1], 3))
  # clt.fit(image)

  # hist =centroid_histogram(clt)

  # print(hist)
  # print(clt.cluster_centers_)
  # center=clt.cluster_centers_

  # bar = plot_colors(hist, clt.cluster_centers_)

  # plt.figure()
  # plt.axis("off")
  # plt.imshow(bar)
  # plt.show()

  # lables=clt.predict(img_eq.reshape((img_eq.shape[0] * img_eq.shape[1], 3)))

  # lables=lables.reshape((img_eq.shape[0] ,img_eq.shape[1]))
  # show_images([(lables==0)*255,(lables==1)*255,(lables==2)*255])

  return [OCR,image_finger]

def flip_orientation3(img):
    '''
    Adjust Orientation of the hand Horizontally
    Still need to modify it vertically ***


    img:Binary image
    '''

    # Flip To make Same Orientation of the hand [Horizontally]
    sum_cols = np.sum(img,axis=0)

    # # print(np.shape(sum_cols))
    # # print(np.shape(sum_cols)[0]%4)
    # remain=np.zeros((np.shape(sum_cols)[0]%2,),np.uint8)
    # # print(np.shape(remain))

    # sum_cols=np.concatenate((sum_cols, remain), axis=0)
    # print(np.shape(sum_cols))

    # # print(np.shape(sum_cols.reshape(4,-1)))
    # sum_cols=sum_cols.reshape(2,-1)
    # sum_cols = np.sum(sum_cols,axis=0)
    # sum_cols=sum_cols.reshape(-1,1)
    # print(np.shape(sum_cols))    


    # 
    OCR=sum_cols
    res=list(compress(range(len(sum_cols==np.max(OCR))),sum_cols==np.max(OCR)))

    if(res[0]<(np.shape(img)[1]/2)):
        img=cv2.flip(img,1) #1=horizontally
        res[0]=np.shape(img)[1]-res[0]

    #res is x with max sum
    return OCR,res[0],img 