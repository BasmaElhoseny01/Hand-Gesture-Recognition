from utils import *
# def equalize_this(image_src, with_plot=False, gray_scale=False):
#     # image_src = read_this(image_file=image_file, gray_scale=gray_scale)
#     if not gray_scale:
#         r_image, g_image, b_image = cv2.split(image_src)

#         r_image_eq = cv2.equalizeHist(r_image)
#         g_image_eq = cv2.equalizeHist(g_image)
#         b_image_eq = cv2.equalizeHist(b_image)

#         image_eq = cv2.merge((r_image_eq, g_image_eq, b_image_eq))
#         cmap_val = None
#     else:
#         image_eq = cv2.equalizeHist(image_src)
#         cmap_val = 'gray'

#     # if with_plot:
#     #     fig = plt.figure(figsize=(10, 20))

#     #     ax1 = fig.add_subplot(2, 2, 1)
#     #     ax1.axis("off")
#     #     ax1.title.set_text('Original')
#     #     ax2 = fig.add_subplot(2, 2, 2)
#     #     ax2.axis("off")
#     #     ax2.title.set_text("Equalized")

#     #     ax1.imshow(image_src, cmap=cmap_val)
#     #     ax2.imshow(image_eq, cmap=cmap_val)
#     #     return True
#     return image_eq


def remove_shadow(img):


    return None

def skin_detection(img):
    '''
    Convert bgr img to Binary with only the hand in it
    @parm img:BGR img

    @return: binarized img with the hand white and Black Background

    '''

    #converting from gbr to hsv color space
    # img_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    #skin color range for hsv color space 
    HSV_mask = cv2.inRange(img_HSV, (0, 15, 0), (17,170,255)) 
    HSV_mask = cv2.morphologyEx(HSV_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))

    
    #converting from gbr to YCbCr color space
    img_YCrCb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    #skin color range for hsv color space 
    YCrCb_mask = cv2.inRange(img_YCrCb, (0, 135, 85), (255,180,135)) 
    YCrCb_mask = cv2.morphologyEx(YCrCb_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))

    #merge skin detection (YCbCr and hsv)
    global_mask=cv2.bitwise_and(YCrCb_mask,HSV_mask)
    global_mask=cv2.medianBlur(global_mask,3)
    global_mask = cv2.morphologyEx(global_mask, cv2.MORPH_OPEN, np.ones((4,4), np.uint8))

    
    HSV_result = cv2.bitwise_not(HSV_mask)
    YCrCb_result = cv2.bitwise_not(YCrCb_mask)
    global_result=cv2.bitwise_not(global_mask)

    # show_images([img_RGB,HSV_result,YCrCb_result,global_result],['Original','HSV','YCrCb','global'])

    return [img_RGB,HSV_result,255-YCrCb_result,global_result]



# img = cv2.imread('./data/Women/2/2_woman (1).JPG')
# equalize_this(img, with_plot=False)


# Read Image
# img = cv2.imread('./data/Women/2/2_woman (86).JPG')
# img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# r_image, g_image, b_image = cv2.split(img_rgb)

# r_image_eq = cv2.equalizeHist(r_image)
# g_image_eq = cv2.equalizeHist(g_image)
# b_image_eq = cv2.equalizeHist(b_image)

# image_eq = cv2.merge((r_image_eq, g_image_eq, b_image_eq))

# Convert to Grey SCale
# img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)



# image_eq = cv2.equalizeHist(img_grey)

# show_images([img_rgb,image_eq],['Original','Equalization'])
# img1 = cv2.imread('./data/Women/2/2_woman (86).JPG')
# img_RGB1,HSV_result1,YCrCb_result1,global_result1=binary_hand_img(img1)

# img2 = cv2.imread('./data/Women/2/2_woman (105).JPG')
# img_RGB2,HSV_result2,YCrCb_result2,global_result2=binary_hand_img(img2)

# img3 = cv2.imread('./data/Women/2/2_woman (126).JPG')
# img_RGB3,HSV_result3,YCrCb_result3,global_result3=binary_hand_img(img3)

# img4 = cv2.imread('./data/Women/2/2_woman (120).JPG')
# img_RGB4,HSV_result4,YCrCb_result4,global_result4=binary_hand_img(img4)

# show_images([img_RGB1,HSV_result1,YCrCb_result1,global_result1,img_RGB2,HSV_result2,YCrCb_result2,global_result2,img_RGB3,HSV_result3,YCrCb_result3,global_result3,img_RGB4,HSV_result4,YCrCb_result4,global_result4])
img1 = cv2.imread('./data/Women/2/2_woman (44).JPG')
# img_grey1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
# imgHist_1 = histogram(img_grey1, nbins=256)

# find frequency of pixels in range 0-255
histr = cv2.calcHist([img1],[0],None,[256],[0,256])
  
# show the plotting graph of an image
plt.plot(histr)
plt.show()

# show_images([img_grey1])
# showHist(img_grey1)

# img_RGB1,HSV_result1,YCrCb_result1,global_result1=skin_detection(img1)

# img2 = cv2.imread('./data/men/2/2_men (11).JPG')
# img_RGB2,HSV_result2,YCrCb_result2,global_result2=skin_detection(img2)

# img3 = cv2.imread('./data/men/2/2_men (15).JPG')
# img_RGB3,HSV_result3,YCrCb_result3,global_result3=skin_detection(img3)
# 

# show_images([img_RGB1,HSV_result1,YCrCb_result1,global_result1,img_RGB2,HSV_result2,YCrCb_result2,global_result2,img_RGB3,HSV_result3,YCrCb_result3,global_result3])


# # Read Image
# img = cv2.imread('./data/Women/2/2_woman (2).JPG')
# img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# # Convert to Grey SCale
# img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)



# image_eq = cv2.equalizeHist(img_grey)

# show_images([img_rgb,img_grey,image_eq],['Original','Grey','Equalization'])
# binary_hand_img(image_eq)


# # Read Image
# img = cv2.imread('./data/Women/2/2_woman (3).JPG')
# img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# # Convert to Grey SCale
# img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)



# image_eq = cv2.equalizeHist(img_grey)

# show_images([img_rgb,img_grey,image_eq],['Original','Grey','Equalization'])
# binary_hand_img(image_eq)


# # Read Image
# img = cv2.imread('./data/Women/2/2_woman (4).JPG')
# img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# # Convert to Grey SCale
# img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)



# image_eq = cv2.equalizeHist(img_grey)

# show_images([img_rgb,img_grey,image_eq],['Original','Grey','Equalization'])
# binary_hand_img(image_eq)





# img = cv2.imread('./data/Women/2/2_woman (2).JPG')
# binary_hand_img(img)

# img = cv2.imread('./data/Women/2/2_woman (3).JPG')
# binary_hand_img(img)

# img = cv2.imread('./data/Women/2/2_woman (4).JPG')
# binary_hand_img(img)