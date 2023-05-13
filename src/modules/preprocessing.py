from utils import *


def preprocessing(images,option, debug=False):
    #SIZE_OF_IMAGE:
    OCR=np.empty((0,256),int)
    classification=[]

    if(option=="0"):
        return images
    for i in range(6):
        index=0
        for img in images[str(i)]:
            if(option=="1"):
                images[str(i)][index] = equalizeS(img, debug)
            elif(option=="2"):
                #OCR
                ocr,_=preprocessing_OCR(img)
                OCR=np.vstack([OCR,ocr])
                classification.append(i)
            else:
                print("Wrong Preprocessing Option!!!",option)
                raise TypeError("Wrong Preprocessing Option")
            index+=1
    if(option=="1"):
        OCR=None
        classification=None
    elif(option=="2"):
        images=None
    return OCR,classification,images



#######################################################################################################
def equalizeS(img, debug=False):
    '''
    - Equalize S
    - Remove Background
    - Applying Mask on the original image BGR
    '''
    # Equalize S
    img_hsv=cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    #Equalize S
    # # img_HSV_eq=np.copy(img_hsv)
    # # img_hsv[:,:,0]=cv2.equalizeHist(img_hsv[:,:,0])
    img_hsv[:,:,1]=cv2.equalizeHist(img_hsv[:,:,1])
    # # img_hsv[:,:,2]=cv2.equalizeHist(img_hsv[:,:,2])

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

    # Extracting skin from the threshold mask  [Debug]
    skin = cv2.bitwise_and(img_hsv, img_hsv, mask=skinMask)
    img_eq=cv2.cvtColor(skin,cv2.COLOR_HSV2BGR) #Extracted Colored Hand
    
    return img_eq


#######################################################################################################
# Doesn't need feature extraction
def preprocessing_OCR(img):
    '''
    1.Get RGB Mask
    2.Flip Horizontal
    3.Translate
    img:BGR
    '''
    #Get Mask
    hand_mask=RGB_Mask(img)

    #Flip
    # OCR,hand_mask=flip_orientation(hand_mask)
    OCR,max_x_ind,min_x_ind,flipped_img=flip_horizontal(hand_mask)
    
    #Translate
    hand_center_x=max_x_ind
    tx=np.shape(flipped_img)[1]-hand_center_x
    translation_matrix=np.array([
        [1,0,tx],
        [0,1,1]
    ],dtype=np.float32)

    flipped_img=flipped_img.astype(np.float32)
    flipped_img=cv2.warpAffine(src=flipped_img,M=translation_matrix,dsize=(np.shape(flipped_img)[1],np.shape(flipped_img)[0]))

    return OCR,flipped_img


##########################################################################
# UTILITIES
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

def flip_horizontal(img,debug=False):
    '''
    img:Binary image
    Adjust Orientation of the hand Horizontally
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
        OCR=np.flip(OCR)
      


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