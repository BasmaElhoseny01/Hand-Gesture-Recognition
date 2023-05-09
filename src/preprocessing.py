from utils import *
import utils


def preprocessing(img, name="", debug=False, gamma=False, close=False):
    '''
    @parm img:BGR Scale img
    '''
    if (gamma):
        img = gamma_trans(img, 4)  # WORKING

    #Skin Mask
    skin = skin_masks(img, name, debug)

    if (close):
        kernel = np.ones((5, 5), np.uint8)
        skin = cv2.morphologyEx(skin, cv2.MORPH_CLOSE, kernel, iterations=10)
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


def ditch_specular(img_bgr):
    # Convert the image to grayscale
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # Apply a median blur to reduce noise
    blur = cv2.medianBlur(gray, 3)
    # Apply a threshold to obtain the specular mask
    # _, mask = cv2.threshold(blur, 240, 255, cv2.THRESH_BINARY_INV)
    # Perform morphological operations to remove small objects and fill holes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 10))
    mask = cv2.morphologyEx(blur, cv2.MORPH_GRADIENT, kernel)
    # Apply the mask to the original image to remove the specular component
    result = cv2.bitwise_and(img_bgr, img_bgr, mask=mask)
    return result


def clahe(img_bgr):
    grayscale = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # Apply contrast limiting adaptive histogram equalization
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(5, 50))
    cl1 = clahe.apply(grayscale)
    # Apply a threshold to obtain the specular mask
    _, mask = cv2.threshold(cl1, 240, 255, cv2.THRESH_BINARY_INV)
    # Perform morphological operations to remove small objects and fill holes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 15))
    mask = cv2.morphologyEx(cl1, cv2.MORPH_OPEN, kernel)

    # Apply the mask to the original image to remove the specular component
    result = cv2.bitwise_and(img_bgr, img_bgr, mask=mask)
    return result


def gamma_trans(img, gamma):
    gamma_table = [np.power(x/255.0, gamma)*255.0 for x in range(256)]
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
    return cv2.LUT(img, gamma_table)


def detect_exposure(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hist, bins = np.histogram(img_gray, bins=256, range=(0, 255))
    print(hist)
