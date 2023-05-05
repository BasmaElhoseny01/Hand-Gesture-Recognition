from utils import *
import utils


def preprocessing(img, name="", debug=False):
    '''
    @parm img:BGR Scale img
    '''
    skin = skin_masks(img, name, debug)

    # Find Contours
    contours, hierarchy = cv2.findContours(
        skin, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    # Draw for debug
    if (debug):
        img_contours = np.copy(img)
        cv2.drawContours(img_contours, contours, -1, (0, 255, 0), 3)

    # Get Largest Contour
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
    largest_contour = sorted_contours[0]
    if (debug):
        largest_contour_img = np.copy(img)
        cv2.drawContours(largest_contour_img,
                         largest_contour, -1, (255, 0, 0), 10)

    # Binary_img_contours[Result]
    hand_contour = np.zeros((np.shape(img)[0], np.shape(img)[1], 1))
    cv2.drawContours(hand_contour, largest_contour, -1, 255, 10)

    if (debug):
        show_images([cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                    skin, img_contours, largest_contour_img, hand_contour],
                    ['RGB'+name, 'Mask(3)YCrCb', 'Contours', 'Largest Contour', 'hand_contour'])

    return hand_contour


def remove_shadow(img):
    """
    Removes shadow form bgr image based on LAB color space

    @param img:BGR image

    @return BGR Image without shadow
    """
    # Convert to Gray Scale

    img_Gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    shadow = img_Gray < 150

    show_images([img, img_Gray, shadow], ['BGR', 'Gray', 'Shadow 150'])
    return None


def skin_masks(img_bgr, name="", debug=False):
    '''
    Gets different masks for the skin color on the image
    @parm img_bgr:BGR Scale img

    @return: Binary Image of Skin Area
    '''
    # HSV Space
    # img_HSV = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    # HSV_mask = cv2.inRange(img_HSV, (0, 15, 0), (17, 170, 255))

    # HSV_mask = cv2.morphologyEx(HSV_mask, cv2.MORPH_OPEN,
    #                             kernel, iterations=3)

    # YCrCb Space
    img_YCrCb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)

    # min_YCrCb = np.array([0, 133, 77], np.uint8)
    # max_YCrCb = np.array([235, 173, 127], np.uint8)
    # imageYCrCb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCR_CB)
    # skinRegionYCrCb = cv2.inRange(imageYCrCb, min_YCrCb, max_YCrCb)
    # skinRegionYCrCb = cv2.morphologyEx(skinRegionYCrCb, cv2.MORPH_OPEN,
    #                                    kernel, iterations=3)

    # YCrCb Space 2
    YCrCb_mask = cv2.inRange(img_YCrCb, (0, 135, 85), (255, 180, 135))
    kernel = np.ones((3, 3), np.uint8)
    YCrCb_mask = cv2.morphologyEx(YCrCb_mask, cv2.MORPH_OPEN,
                                  kernel, iterations=3)

    # show_images([cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), HSV_mask, skinRegionYCrCb,
    #             YCrCb_mask, ], ['RGB'+name, 'Mask(1) HSV', 'Mask(2)YCrCb', 'Mask(3)YCrCb'])
    # if(debug):
    # show_images([cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB),
    #             YCrCb_mask, ], ['RGB'+name, 'Mask(3)YCrCb'])

    return YCrCb_mask
