# Utils for preprocessing
from utils import *

def calculate_shadow_mask(org_image: np.ndarray,
                          ab_threshold: int,
                          region_adjustment_kernel_size: int) -> np.ndarray:
    '''
    Function to calculate shadow from the image from the image colors so to be able to remove it
    '''
    # Convert the L,A,B from 0 to 255 to 0 - 100 and -128 - 127 and -128 - 127 respectively
    lab_img = cv2.cvtColor(org_image, cv2.COLOR_BGR2LAB)
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


def equalizeV(img_hsv):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    H, S, V = cv2.split(img_hsv)
    v_equalized = clahe.apply(V)
    h_new = cv2.subtract(H, H)
    equalized = cv2.merge([h_new, S, v_equalized])
    return equalized


def shadow_remove(img):
    '''
    function to remove the shadow from the image
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


def cut_fingers(hand_binary, hand_center_x, raduis=150):
    '''
    Function to cut fingers from the image by drawing circle from the center so we have fingers separated
    '''
    image_finger = cv2.circle(
        hand_binary, (hand_center_x, np.shape(hand_binary)[0]//2), raduis, 0, -1)
    return image_finger


def gammaCorrection(src, gamma):
    '''
    Apply Gamma Correction preprocessing on  the image :D
    '''
    invGamma = 1/gamma
    table = [((i/255)**invGamma)*255 for i in range(256)]
    table = np.array(table, np.uint8)
    return cv2.LUT(src, table)


def YCrCb_Mask(img):
    '''
    get YCrCb of the image
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
