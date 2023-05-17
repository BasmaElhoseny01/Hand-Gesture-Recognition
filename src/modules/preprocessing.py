from global_ import IMG_SIZE

from utils import *


def preprocessing(images, option, debug=False):
    # SIZE_OF_IMAGE:
    OCR = np.empty((0, IMG_SIZE[0]), int)
    classification = []
    hands = {'0': None, '1': None, '2': None,
             '3': None, '4': None, '5': None}

    if (option == "0"):
        return None, None, images
    for i in range(6):
        index = 0
        j_write = 0
        for img in images[str(i)]:
            if (option == "1"):
                images[str(i)][index] = equalizeS(img, debug)
            elif (option == "2"):
                # OCR
                ocr, _ = preprocessing_OCR(img)
                OCR = np.vstack([OCR, ocr])
                classification.append(i)
            elif (option == "3"):
                _, _, img_hand = hand_shadow_based_preprocessing(img, debug)
                if (hands[str(i)] is None):
                    hands[str(i)] = np.array([img_hand])  # 1* 128*256
                else:
                    # print('hands[str(i)]',np.shape(hands[str(i)]))
                    img_hand = np.atleast_3d(img_hand)  # 128*286*1
                    # -> swap axes to be 1*128*286  2<->0
                    img_hand = np.moveaxis(img_hand, [2], [0])
                    # print('img_hand',np.shape(img_hand))
                    hands[str(i)] = np.append(hands[str(i)], img_hand, axis=0)

            elif (option == "4"):
                _, image_finger = cut_fingers_preprocessing(img, debug)
                if (hands[str(i)] is None):
                    hands[str(i)] = np.array([image_finger])  # 1* 128*256
                else:
                    # print('hands[str(i)]',np.shape(hands[str(i)]))
                    image_finger = np.atleast_3d(image_finger)  # 128*286*1
                    # -> swap axes to be 1*128*286  2<->0
                    image_finger = np.moveaxis(image_finger, [2], [0])
                    # print('img_hand',np.shape(img_hand))
                    hands[str(i)] = np.append(
                        hands[str(i)], image_finger, axis=0)
            elif (option == "5"):
                img_grey = Grey_Scale_Preprocessing(img)
                if (hands[str(i)] is None):
                    hands[str(i)] = np.array([img_grey])  # 1* 128*256
                else:
                    # print('hands[str(i)]',np.shape(hands[str(i)]))
                    img_grey = np.atleast_3d(img_grey)  # 128*286*1
                    # -> swap axes to be 1*128*286  2<->0
                    img_grey = np.moveaxis(img_grey, [2], [0])
                    # print('img_hand',np.shape(img_hand))
                    hands[str(i)] = np.append(hands[str(i)], img_grey, axis=0)
            elif (option == "shadow"):
                # BGR image :D
                images[str(i)][index] = remove_shadow_Ycrcb(img)
            elif (option == "ycrcb"):
                images[str(i)][index] = YCrCb(img)
            elif(option=="6"):
                s_channel=sChannelPreprocessing(img)
                if(hands[str(i)] is None):
                    hands[str(i)]=np.array([s_channel]) #1* 128*256
                else:
                    # print('hands[str(i)]',np.shape(hands[str(i)]))
                    s_channel=np.atleast_3d(s_channel)#128*286*1 
                    s_channel=np.moveaxis(s_channel,[2],[0])#-> swap axes to be 1*128*286  2<->0
                    # print('img_hand',np.shape(img_hand))
                    hands[str(i)]=np.append(hands[str(i)],s_channel,axis=0)
            elif(option=="yasmine"):
                 images[str(i)][index]=preprocessing_yasmine(img)
            else:
                print("Wrong Preprocessing Option!!!", option)
                raise TypeError("Wrong Preprocessing Option")
            index += 1

        if (option not in ["shadow", "ycrcb","yasmine"]):
            images[str(i)] = None  # Deallocate for Memory

    if (option == "1"):
        OCR = None
        classification = None
    elif (option == "2"):
        images = None
    elif (option == "3" or option == "4" or option == "5" or option == "6"):
        images = hands

    return OCR, classification, images




##############################################################################################################################
def preprocessing_yasmine(img, debug=False,path=""):

    # Shadow removal
    shadow_mask = calculate_shadow_mask(org_image=img, ab_threshold=0, region_adjustment_kernel_size=10)
    shadow_removed = img
    shadow_removed[shadow_mask == 255] = 0

    # convert image from BGR to HSV and GRAYSCALE
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _,gray_threshold = cv2.threshold(img_gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # equalize v channel hsv 
    # img_hsv_equalized = equalizeV(img_hsv)

    # extract the S channel from HSV
    s_channel = img_hsv[:,:,1]
    
    # extract the V channel from HSV
    v_channel = img_hsv[:,:,2]
    
    # get threshold for each image seperately
    s_threshold = getThreshold(s_channel)
    
    # get threshold for each image seperately
    v_threshold = getThreshold(v_channel)
    

    # apply threshhold on image
    s_thresholded =  cv2.inRange(s_channel, s_threshold, 255)
    
    # Try OTSU thresholding
    _,otsu_s_threshhold = cv2.threshold(s_channel, 0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    # apply threshhold on image
    v_thresholded =  cv2.inRange(v_channel, v_threshold, 255)
    
    # Region Filling 
    # fill empty regions if hand mask
    # img_fill=s_thresholded.copy()
    # h,w=s_thresholded.shape[:2]
    # mask=np.zeros((h+2,w+2),np.uint8)

    # cv2.floodFill(img_fill,mask,(0,0),255) #img_fill marks regions filled -> not so that we can see it bec they are black
    # region_filling=cv2.bitwise_not(mask)
    # region_filling[region_filling == 255] = 255
    # region_filling[region_filling == 254] = 0

    # # resize region filled mask to be compatable with original image size
    # region_filling = region_filling[0:img_gray.shape[0], 0:img_gray.shape[1]]

    # and the s and v channels 
    s_and_v = cv2.bitwise_and(s_thresholded, v_thresholded)    

    if(np.sum(s_and_v)//255 < 0.25*s_and_v.shape[0]*s_and_v.shape[1]):
        s_and_v = s_thresholded
            
    # Closing
    kernel = np.ones((3, 3), np.uint8)
    edge_closing = cv2.morphologyEx(s_and_v, cv2.MORPH_CLOSE, kernel, iterations=2)

    _, _, _, mask_flipped, original_fliped = flip_horizontal(img=edge_closing, Original=img)
    
    img_anded = cv2.bitwise_and(original_fliped, original_fliped, mask=mask_flipped)
    
    _,result_threshold = cv2.threshold(cv2.cvtColor(img_anded, cv2.COLOR_BGR2GRAY),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)


    if(debug):
        # show_images([s_channel, v_channel, s_thresholded, v_thresholded, region_filling, img_ored, s_and_vinv],['s_channel', 'v_channel','s_thresholded', 'v_thresholded' , 'region_filling', 'oring', 's_and_vinv'])
        show_images([gray_threshold, result_threshold, otsu_s_threshhold], ['gray_threshold', 'result_threshold', 'otsu_s_threshold'])
        show_images([img, shadow_removed,s_channel, v_channel, s_thresholded, v_thresholded, s_and_v, edge_closing, img_anded],['original', 'shadow_Removed','s_channel', 'v_channel','s_thresholded', 'v_thresholded' , 's_and_v', 'closing', 'img_anded'],save=False,path_save=path)
 
    return img_anded
##############################################################################################################################

def S_channel(img, debug=False, path=""):
    ''''
    - 
    -
    -

    @param img:bgr img
    '''

    if (debug):
        show_images([], [], save=True, path_save=path)

    return None

##############################################################################################################################


def YCrCb(img, debug=False):
    ''''
    - Get Mask of YCrCb
    - Flip
    - And Mask with BGR image

    @param img:bgr img
    '''

    # ---------------------------------------------------------------------------------------------------------------
    # YCRCB
    YCrCb_mask = YCrCb_Mask(img)
    return YCrCb_mask #Temp

    # ---------------------------------------------------------------------------------------------------------------
    # Flip
    if (debug):
        flipped_original = np.copy(img)
    else:
        flipped_original = img
    _, _, _, flipped_img, flipped_original = flip_horizontal(
        YCrCb_mask, debug=False, Original=flipped_original)

    # --------------------------------------------------------------------------------------------------------------
    # Mask Anding
    masked_img = cv2.bitwise_and(
        flipped_original, flipped_original, mask=flipped_img)

    return masked_img

##############################################################################################################################


def remove_shadow_Ycrcb(img, debug=False):
    '''
    - Replace shadow area by black
    - Get Mask of YCrCb
    - Flip
    - And Mask with BGR image
    '''
    # Shadow removal
    shadow_mask = calculate_shadow_mask(
        org_image=img, ab_threshold=0, region_adjustment_kernel_size=10)
    shadow_removed = img
    shadow_removed[shadow_mask == 255] = 0

    # ---------------------------------------------------------------------------------------------------------------
    # YCRCB
    YCrCb_mask = YCrCb_Mask(shadow_removed)

    # ---------------------------------------------------------------------------------------------------------------
    # Flip
    if (debug):
        flipped_original = np.copy(img)
    else:
        flipped_original = img
    _, _, _, flipped_img, flipped_original = flip_horizontal(
        YCrCb_mask, debug=False, Original=flipped_original)

    # --------------------------------------------------------------------------------------------------------------
    # Mask Anding
    masked_img = cv2.bitwise_and(
        flipped_original, flipped_original, mask=flipped_img)
    return masked_img

##############################################################################################################################


def equalizeS(img, debug=False):
    '''
    - Equalize S
    - Remove Background
    - Flip  xxxx (3)
    - Applying Mask on the original image BGR
    '''
    # Equalize S
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Equalize S
    # # img_HSV_eq=np.copy(img_hsv)
    # # img_hsv[:,:,0]=cv2.equalizeHist(img_hsv[:,:,0])
    img_hsv[:, :, 1] = cv2.equalizeHist(img_hsv[:, :, 1])
    # # img_hsv[:,:,2]=cv2.equalizeHist(img_hsv[:,:,2])

    # Remove BG
    # Defining HSV Thresholds
    lower_threshold = np.array([0, 48, 80], dtype=np.uint8)
    upper_threshold = np.array([20, 255, 255], dtype=np.uint8)

    # Single Channel mask,denoting presence of colours in the about threshold
    skinMask_b = cv2.inRange(img_hsv, lower_threshold, upper_threshold)//255

    # Cleaning up mask using Gaussian Filter
    skinMask = cv2.GaussianBlur(skinMask_b, (3, 3), 0)

    # Erosion
    kernel = np.ones((3, 3), np.uint8)
    skinMask = cv2.morphologyEx(
        skinMask, cv2.MORPH_ERODE, kernel, iterations=4)  # erode

    # #Flip
    # # OCR,hand_mask=flip_orientation(hand_mask)
    # OCR,max_x_ind,min_x_ind,flipped_img=flip_horizontal(skinMask)
    # skinMask=flipped_img

    # Extracting skin from the threshold mask
    img_eq = cv2.cvtColor(skin, cv2.COLOR_HSV2BGR)  # Extracted Colored Hand

    return img_eq

#######################################################################################################
def equalizeV(img_hsv):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    H, S, V = cv2.split(img_hsv)
    v_equalized = clahe.apply(V)
    h_new = cv2.subtract(H, H)
    equalized = cv2.merge([h_new, S, v_equalized])
    return equalized
#############################################################################################
# Doesn't need feature extraction
def preprocessing_OCR(img):
    '''
    1.Get RGB Mask
    2.Flip Horizontal
    3.Translate
    img:BGR
    '''
    # Get Mask
    hand_mask = RGB_Mask(img)

    # Flip
    # OCR,hand_mask=flip_orientation(hand_mask)
    OCR, max_x_ind, min_x_ind, flipped_img = flip_horizontal(hand_mask)

    # Translate
    hand_center_x = max_x_ind
    tx = np.shape(flipped_img)[1]-hand_center_x
    translation_matrix = np.array([
        [1, 0, tx],
        [0, 1, 1]
    ], dtype=np.float32)

    flipped_img = flipped_img.astype(np.float32)
    flipped_img = cv2.warpAffine(src=flipped_img, M=translation_matrix, dsize=(
        np.shape(flipped_img)[1], np.shape(flipped_img)[0]))

    return OCR, flipped_img


##########################################################################

def hand_shadow_based_preprocessing(img, debug=False):
    '''
    1.Remove shadow
    2.Gamma Correction for the removal of shadow
    3. YCrCb Mask
    4. And 2&3
    5. Flip

    @param img:bgr

    @return img_flip binary
    '''
    # Change Color
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # --------------------------------------------------------------------------------------------------------
    # Shadow removal
    shadow_removed = shadow_remove(img)

    # ----------------------------------------------------------------------------------------------------------------
    # Gamma Correction
    # shadow_removed_gamma=gammaCorrection(shadow_removed,0.4)
    shadow_removed_gamma = shadow_removed
    shadow_removed_gamma = cv2.cvtColor(
        shadow_removed_gamma, cv2.COLOR_RGB2GRAY)  # Convert it to Gray

    # --------------------------------------------------------------------------------------------------------
    # HSV MASK (Bad)
    # hsv_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    # lower_skin = (0, 20, 70)
    # upper_skin = (20, 255, 255)
    # skin_hsv=cv2.inRange(hsv_image, lower_skin, upper_skin)
    # end = time.time()
    # print('Gamma',end-st)

    # --------------------------------------------------------------------------------------------------------
    # YCrCb Mask
    YCrCb_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    lower_skin = (0, 135, 85)
    upper_skin = (255, 180, 135)
    skin_ycrcb = cv2.inRange(YCrCb_image, lower_skin, upper_skin)

    # Anding Shadow & YCrCb
    region_anding = cv2.bitwise_and(skin_ycrcb, shadow_removed_gamma)  # 0&255

    # Erosion
    kernel = np.ones((4, 4), np.uint8)
    # edge = cv2.morphologyEx(edge, cv2.MORPH_DILATE, kernel, iterations=5) #erode
    edge_dilate = cv2.morphologyEx(
        region_anding, cv2.MORPH_ERODE, kernel, iterations=2)  # erode

    # Flip
    _, max_x_ind, min_x_ind, img_flip = flip_horizontal(edge_dilate, debug)

    return max_x_ind, min_x_ind, img_flip

################################################################################


def cut_fingers_preprocessing(img, debug=False):
    '''
    1.Remove shadow
    2.Gamma Correction for the removal of shadow
    3. YCrCb Mask
    4. And 2&3
    5. Flip
    6. Translate
    7.Cut Fingers
    '''
    max_x_ind, min_x_ind, img_flip = hand_shadow_based_preprocessing(
        img, debug=debug)

    # -------------------------------------------------------------------------
    # Translate
    hand_center_x = max_x_ind
    tx = np.shape(img_flip)[1]-hand_center_x
    translation_matrix = np.array([
        [1, 0, tx],
        [0, 1, 1]
    ], dtype=np.float32)

    img_flip = img_flip.astype(np.float32)
    img_flip_tran = cv2.warpAffine(src=img_flip, M=translation_matrix, dsize=(
        np.shape(img_flip)[1], np.shape(img_flip)[0]))

    # --------------------------------------------------------------------------
    # Cut Fingers  img_flip_tran is by reference
    raduis = (max_x_ind-min_x_ind)//2
    image_finger = cut_fingers(
        img_flip_tran, np.shape(img_flip_tran)[1]-1, raduis)
    image_finger = np.uint8(image_finger*255)

    # Thresholding
    # ret2,th2 = cv2.threshold(image_finger,150,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    ret2, image_finger = cv2.threshold(
        image_finger, 10, 255, cv2.THRESH_BINARY)

    return img_flip, image_finger

################################################################################


def Grey_Scale_Preprocessing(img):
    '''
    -Covert to Grey Scale
    '''
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img
################################################################################
def sChannelPreprocessing(img,debug=False):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # extract the S channel from HSV
    S = img_hsv[:,:,1]

    if(debug):
        show_images([S],['s_channel'])
 
    return S

################################################################################

def preprocessing_yasmine(img, debug=False,path=""):

    # Shadow removal
    shadow_mask = calculate_shadow_mask(org_image=img, ab_threshold=0, region_adjustment_kernel_size=10)
    shadow_removed = img
    shadow_removed[shadow_mask == 255] = 0



    # convert image from BGR to HSV and GRAYSCALE
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _,gray_threshold = cv2.threshold(img_gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # equalize v channel hsv 
    # img_hsv_equalized = equalizeV(img_hsv)

    # print(1)
    # extract the S channel from HSV
    s_channel = img_hsv[:,:,1]
    # print(2)

    # extract the V channel from HSV
    v_channel = img_hsv[:,:,2]
    # print(3)

    # get threshold for each image seperately
    s_threshold = getThreshold(s_channel)
    # print(4)

    # get threshold for each image seperately
    v_threshold = getThreshold(v_channel)
    # print(5)


    # apply threshhold on image
    s_thresholded =  cv2.inRange(s_channel, s_threshold, 255)
    # print(6)

    # TRY OTSU
    _,otsu_s_threshhold = cv2.threshold(s_channel, 0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    # apply threshhold on image
    v_thresholded =  cv2.inRange(v_channel, v_threshold, 255)
    # print(7)

    # hsv_thresholded = cv2.inRange(img_hsv, (0,s_threshold, v_threshold), (0, 255, 255))

    # Region Filling 
    # fill empty regions if hand mask
    # img_fill=s_thresholded.copy()
    # h,w=s_thresholded.shape[:2]
    # mask=np.zeros((h+2,w+2),np.uint8)

    # cv2.floodFill(img_fill,mask,(0,0),255) #img_fill marks regions filled -> not so that we can see it bec they are black
    # region_filling=cv2.bitwise_not(mask)
    # region_filling[region_filling == 255] = 255
    # region_filling[region_filling == 254] = 0

    # # resize region filled mask to be compatable with original image size
    # region_filling = region_filling[0:img_gray.shape[0], 0:img_gray.shape[1]]

    # and the hand mask with the gray image
    # img = np.asarray(img)
    # b_channel = img[:,:,0]
    # region_filling = np.asarray(region_filling)

    # img_anded = cv2.bitwise_and(img, img, mask=region_filling)
    s_and_v = cv2.bitwise_and(s_thresholded, v_thresholded)
    # print(8)

    # v_inverted = cv2.bitwise_not(v_thresholded)
    # s_and_vinv = cv2.bitwise_and(s_thresholded, v_inverted)
    # img_ored = cv2.bitwise_or(s_thresholded, v_thresholded)

    if(np.sum(s_and_v)//255 < 0.25*s_and_v.shape[0]*s_and_v.shape[1]):
        s_and_v = s_thresholded
        # print(9)
    # print(10)



    # Region Filling 
    # fill empty regions if hand mask
    # img_fill=s_and_v.copy()
    # h,w=s_and_v.shape[:2]
    # mask=np.zeros((h+2,w+2),np.uint8)

    # cv2.floodFill(img_fill,mask,(0,0),255) #img_fill marks regions filled -> not so that we can see it bec they are black
    # region_filling_anded=cv2.bitwise_not(mask)
    # region_filling_anded[region_filling_anded == 255] = 255
    # region_filling_anded[region_filling_anded == 254] = 0

    # Erosion
    kernel = np.ones((3, 3), np.uint8)
    # edge = cv2.morphologyEx(edge, cv2.MORPH_DILATE, kernel, iterations=5) #erode
    edge_closing = cv2.morphologyEx(s_and_v, cv2.MORPH_CLOSE, kernel, iterations=2) # erode

    # print(11)


    # img_anded = cv2.bitwise_and(img, img, mask=edge_closing)

    _, _, _, mask_flipped, original_fliped = flip_horizontal(img=edge_closing, Original=img)
    # print(12)


    img_anded = cv2.bitwise_and(original_fliped, original_fliped, mask=mask_flipped)
    # print(13)

    _,result_threshold = cv2.threshold(cv2.cvtColor(img_anded, cv2.COLOR_BGR2GRAY),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)


    if(debug):
        # show_images([s_channel, v_channel, s_thresholded, v_thresholded, region_filling, img_ored, s_and_vinv],['s_channel', 'v_channel','s_thresholded', 'v_thresholded' , 'region_filling', 'oring', 's_and_vinv'])
        show_images([gray_threshold, result_threshold, otsu_s_threshhold], ['gray_threshold', 'result_threshold', 'otsu_s_threshold'])
        show_images([img, shadow_removed,s_channel, v_channel, s_thresholded, v_thresholded, s_and_v, edge_closing, img_anded],['original', 'shadow_Removed','s_channel', 'v_channel','s_thresholded', 'v_thresholded' , 's_and_v', 'closing', 'img_anded'],save=False,path_save=path)
 
    return img_anded

################################################################################
# UTILITIES


def YCrCb_Mask(img):
    '''
    img:bgr
    '''
    # --------------------------------------------------------------------------------------------------------
    # YCrCb Mask
    YCrCb_image = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    lower_skin = (0, 135, 85)
    upper_skin = (255, 180, 135)
    skin_ycrcb = cv2.inRange(YCrCb_image, lower_skin, upper_skin)
    return skin_ycrcb


def RGB_Mask(img):
    '''
    Get RGB Mask of the Skin
    Ref:https://medium.com/swlh/face-detection-using-skin-tone-threshold-rgb-ycrcb-python-implementation-2d4f62d376f1


    img:BGR
    NB:RGB_Mask(removed_shadow).astype(np.uint8) Convert to unint 8 bit to be able to and it with 8UI 3 channel image
    '''
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]

    BGR_Max = np.maximum.reduce([R, G, B])
    BGR_Min = np.minimum.reduce([R, G, B])

    Rule_1 = np.logical_and.reduce(
        [R > 95, G > 40, B > 20, (BGR_Max-BGR_Min) > 15, abs(R-G) > 15, R > G, R > B])
    Rule_2 = np.logical_and.reduce(
        [R > 220, G > 210, B > 170, abs(R-G) <= 15, R > B, G > B])

    RGB_Rule = np.bitwise_or(Rule_1, Rule_2)
    RGB_Rule = RGB_Rule*255

    return RGB_Rule

def flip_horizontal(img, debug=False, Original=None):
    '''
    img:Binary image
    Adjust Orientation of the hand Horizontally
    '''

    # Compute OCR
    OCR = np.sum(img, axis=0)

    # Get Max index
    res = list(compress(range(len(OCR == np.max(OCR))), OCR == np.max(OCR)))
    max_x_ind = res[0]

    # Get Min index
    # CHECK: Handle inf case
    result = min(enumerate(OCR),
                 key=lambda x: x[1] if x[1] > 20 else float('inf'))
    # print("Position : {}, Value : {}".format(*result))
    min_x_ind = result[0]

    if (min_x_ind > max_x_ind):
        img = cv2.flip(img, 1)  # 1=horizontally
        max_x_ind = np.shape(img)[1]-max_x_ind
        min_x_ind = np.shape(img)[1]-min_x_ind
        OCR = np.flip(OCR)
        if (Original is not None):
            Original_fliped = cv2.flip(Original, 1)
    else:
        Original_fliped = Original

    if (debug):
        print("Image Size", np.shape(img))
        print("OCR shape", np.shape(OCR))
        print("Max x is at", max_x_ind)
        print("Min x is at", min_x_ind)

        # x-coordinates of left sides of bars
        left = range(0, np.shape(OCR)[0])
        print("X Range", left)

        # heights of bars
        height = OCR
        print("Heights", height)

        # plotting a bar chart
        plt.bar(left, height,
                width=0.1, color=['red', 'green'])

        # naming the x-axis
        plt.xlabel('x - axis')
        # naming the y-axis
        plt.ylabel('y - axis')
        # plot title
        plt.title('My bar chart!')

        # function to show the plot
        plt.show()
        show_images([img], ['Flipped'])

    return OCR, max_x_ind, min_x_ind, img, Original_fliped


def shadow_remove(img):
    '''
    img:bgr img
    @return RGB image with shadow assumed to be removed 😂
    '''
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert it to Gray

    rgb_planes = cv2.split(img)
    result_norm_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((2, 2), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 9)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        # print(diff_img)
        norm_img = cv2.normalize(
            diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV2_8UC1)
        result_norm_planes.append(norm_img)
    shadow_removed = cv2.merge(result_norm_planes)

    return shadow_removed


def calculate_shadow_mask(org_image: np.ndarray,
                          ab_threshold: int,
                          region_adjustment_kernel_size: int) -> np.ndarray:
    lab_img = cv2.cvtColor(org_image, cv2.COLOR_BGR2LAB)

    # Convert the L,A,B from 0 to 255 to 0 - 100 and -128 - 127 and -128 - 127 respectively
    l_range = (0, 100)
    ab_range = (-128, 127)

    lab_img = lab_img.astype('int16')
    lab_img[:, :, 0] = lab_img[:, :, 0] * l_range[1] / 255
    lab_img[:, :, 1] += ab_range[0]
    lab_img[:, :, 2] += ab_range[0]

    # Calculate the mean values of L, A and B across all pixels
    means = [np.mean(lab_img[:, :, i]) for i in range(3)]
    thresholds = [means[i] - (np.std(lab_img[:, :, i]) / 3) for i in range(3)]

    # Apply threshold using only L
    if sum(means[1:]) <= ab_threshold:
        mask = cv2.inRange(lab_img, (l_range[0], ab_range[0], ab_range[0]),
                           (thresholds[0], ab_range[1], ab_range[1]))
    else:  # Else, also consider B channel
        mask = cv2.inRange(lab_img, (l_range[0], ab_range[0], ab_range[0]),
                           (thresholds[0], ab_range[1], thresholds[2]))

    kernel_size = (region_adjustment_kernel_size,
                   region_adjustment_kernel_size)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, mask)
    cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, mask)

    return mask


def cut_fingers(hand_binary, hand_center_x, raduis=150):
    image_finger = cv2.circle(
        hand_binary, (hand_center_x, np.shape(hand_binary)[0]//2), raduis, 0, -1)
    return image_finger


def gammaCorrection(src, gamma):
    invGamma = 1/gamma
    table = [((i/255)**invGamma)*255 for i in range(256)]
    table = np.array(table, np.uint8)
    return cv2.LUT(src, table)



###################################################################################################################

def sChannelPreprocessing(img,debug=False):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # extract the S channel from HSV
    S = img_hsv[:,:,1]

    if(debug):
        show_images([S],['s_channel'])
 
    return S

# def clahe(img_bgr):
#     grayscale = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
#     # Apply contrast limiting adaptive histogram equalization
#     clahe = cv2.createCLAHE(clipLimit=20.0, tileGridSize=(5, 50))
#     cl1 = clahe.apply(grayscale)
#     # Apply a threshold to obtain the specular mask
#     _, mask = cv2.threshold(cl1, 240, 255, cv2.THRESH_BINARY_INV)
#     # Perform morphological operations to remove small objects and fill holes
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 15))
#     mask = cv2.morphologyEx(cl1, cv2.MORPH_OPEN, kernel)

#     # Apply the mask to the original image to remove the specular component
#     result = cv2.bitwise_and(img_bgr, img_bgr, mask=mask)
#     return result

# def preprocessing_yasmine_basma(img, debug=False,path=""):

#     img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#     # # convert image from BGR to HSV and GRAYSCALE

#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#     equalized = clahe.apply(img_hsv[:,:,2])



#     # result=cv2.merge([img_hsv[:,:,0],img_hsv[:,:,1],equalized])
#     # result=cv2.cvtColor(result, cv2.COLOR_HSV2BGR)



#     # Otsu's thresholding
#     ret2,th2 = cv2.threshold(equalized,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

#     # Erosion
#     kernel = np.ones((4, 4), np.uint8)
#     # edge = cv2.morphologyEx(edge, cv2.MORPH_DILATE, kernel, iterations=5) #erode
#     edge_dilate = cv2.morphologyEx(
#         th2, cv2.MORPH_DILATE, kernel, iterations=3)  # erode
    

#     # Mask Anding
#     masked_img = cv2.bitwise_and(
#         img, img, mask=edge_dilate)
#     masked_img[masked_img==0]=255
    
#     masked_img = cv2.cvtColor(masked_img, cv2.COLOR_BGR2RGB)


#     ycrcb=YCrCb(cv2.cvtColor(masked_img, cv2.COLOR_RGB2BGR))

    
#     # Region Filling 
#     # fill empty regions if hand mask
#     img_fill=ycrcb.copy()
#     h,w=ycrcb.shape[:2]
#     mask=np.zeros((h+2,w+2),np.uint8)

#     cv2.floodFill(img_fill,mask,(0,0),255) #img_fill marks regions filled -> not so that we can see it bec they are black
#     region_filling=cv2.bitwise_not(mask)
#     # region_filling[region_filling == 255] = 255
#     # region_filling[region_filling == 254] = 0



#     show_images([img_hsv[:,:,0],img_hsv[:,:,1],equalized,th2,edge_dilate,masked_img,ycrcb,region_filling],save=False,path_save=path)



#     # #Shadow
#     # shadow_mask = calculate_shadow_mask(
#     #     org_image=result, ab_threshold=0, region_adjustment_kernel_size=10)
#     # shadow_removed = result
#     # shadow_removed[shadow_mask == 255] = 0


#     # ycrcb=YCrCb(shadow_removed)

#     # show_images([cv2.cvtColor(img, cv2.COLOR_BGR2RGB),cv2.cvtColor(result, cv2.COLOR_BGR2RGB),ycrcb_or,ycrcb],(['','','Ycold','ycnew']))


#     # show_images([cv2.cvtColor(img, cv2.COLOR_BGR2RGB),img_hsv[:,:,0],img_hsv[:,:,1],img_hsv[:,:,2],result])
#     return None
#     img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#     # extract the S channel from HSV
#     S = img_hsv[:,:,1]


#     # get threshold for each image seperately
#     threshold = getThreshold(S)
    
#     # apply threshhold on image
#     s_threshold =  cv2.inRange(S, threshold, 255)


#     # Region Filling 
#     # fill empty regions if hand mask
#     img_fill=s_threshold.copy()
#     h,w=s_threshold.shape[:2]
#     mask=np.zeros((h+2,w+2),np.uint8)

#     cv2.floodFill(img_fill,mask,(0,0),255) #img_fill marks regions filled -> not so that we can see it bec they are black
#     region_filling=cv2.bitwise_not(mask)
#     region_filling[region_filling == 255] = 255
#     region_filling[region_filling == 254] = 0

#     # resize region filled mask to be compatable with original image size
#     region_filling = region_filling[0:img_gray.shape[0], 0:img_gray.shape[1]]


#     ycrcb=YCrCb(img)
#     img_flip=remove_shadow_Ycrcb(img)
#     show_images([cv2.cvtColor(img, cv2.COLOR_BGR2RGB),s_threshold,ycrcb],['Original','s_threshold'])


#     return None


#     # and the hand mask with the gray image
#     img_anded = cv2.bitwise_and(img_gray, region_filling)
#     # img_anded = cv2.cvtColor(img_anded, cv2.COLOR_GRAY2BGR)


#     if(debug):
#         print(np.max(region_filling))
#         print(np.min(region_filling))
#         print(img_gray.shape)
#         print(region_filling.shape)
#         print(threshold)
#         show_images([S, s_threshold, region_filling, img_anded],['s_channel', 'thresholded', 'filled_regions', 'image_anded'],save=True, path_save=path)
 
#     return img_anded
