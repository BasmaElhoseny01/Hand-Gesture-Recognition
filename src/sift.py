# # # # IMPORTS
# import cv2
# # # import utils

# # # # Loading the image
# # # img = cv2.imread('./data/Women/2/2_woman (14).JPG')
# # # img2 = cv2.imread('./data/Women/2/2_woman (17).JPG')
 
# # #  # Converting image to grayscale
# # # gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# # # gray2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)


# # # # Applying SIFT detector
# # # sift = cv2.xfeatures2d.SIFT_create()

# # # #Extract Features
# # # kp = sift.detect(gray, None)
# # # kp2 = sift.detect(gray2, None)

 
# # # # Marking the keypoint on the image using circles
# # # img = cv2.drawKeypoints(gray, kp, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# # # img2 = cv2.drawKeypoints(gray2, kp, img2, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


# # # utils.show_images([img,img2])


# # # importing required libraries of opencv
# from utils import *
# # import cv2
  
# # # importing library for plotting
# # from matplotlib import pyplot as plt
  
# # # reads an input image
# # # img = cv2.imread('./data/Women/2/2_woman (14).JPG',0)
# # image1 = cv2.imread('./data/Women/2/2_woman (14).JPG')
# # image2 = cv2.imread('./data/Women/2/2_woman (15).JPG')


# # img = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
# # img2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)


# # # applying Otsu thresholding
# # # as an extra flag in binary 
# # # thresholding     
# # ret, thresh1 = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY + 
# #                                             cv2.THRESH_OTSU) 

# # ret2, thresh2 = cv2.threshold(img2, 100, 255, cv2.THRESH_BINARY + 
# # cv2.THRESH_OTSU) 
# # print(ret)

# # show_images([img,thresh1,img2,thresh2],['Original','Otsuo Thres','org','thres2'])


# img = cv2.imread('./data/Women/1/1_woman (29).JPG')

# # Create single channel greyscale for thresholding
# # myimage_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
  
# # initializing subtractor 
# fgbg = cv2.bgsegm.createBackgroundSubtractorGMG()

# fgmask = fgbg.apply(img)
  
# fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)  

# show_images([img,fgmask])


# # ret,thresh1 = cv2.threshold(myimage_grey,150,255,cv2.THRESH_BINARY_INV)
# # # remove=thresh1+myimage_grey
# # # print(np.max(remove))

# # remove=np.copy(myimage_grey)
# # remove[remove<150]=0
# # show_images([myimage_grey,thresh1,remove])