from utils import *

def gammaCorrection(img, gamma):
    '''
    By adjusting the gamma value we can remove the darkness of an image.
    @parm img: Source Image
    @parm gamma: Gamma Value of Correction

    @return: Corrected image
    '''
    invGamma = 1 / gamma

    table = [((i / 255) ** invGamma) * 255 for i in range(256)]
    table = np.array(table, np.uint8)

    return cv2.LUT(img, table)

# Obtaining the Bright and Dark channel Prior
def get_illumination_channel(I, w):
    M, N, _ = I.shape
    # padding for channels
    padded = np.pad(I, ((int(w/2), int(w/2)), (int(w/2), int(w/2)), (0, 0)), 'edge')
    darkch = np.zeros((M, N))
    brightch = np.zeros((M, N))
 
    for i, j in np.ndindex(darkch.shape):
        darkch[i, j] = np.min(padded[i:i + w, j:j + w, :]) # dark channel
        brightch[i, j] = np.max(padded[i:i + w, j:j + w, :]) # bright channel
 
    return darkch, brightch

# Computing Global Atmosphere Lighting
def get_atmosphere(I, brightch, p=0.1):
    M, N = brightch.shape
    flatI = I.reshape(M*N, 3) # reshaping image array
    flatbright = brightch.ravel() #flattening image array
 
    searchidx = (-flatbright).argsort()[:int(M*N*p)] # sorting and slicing
    A = np.mean(flatI.take(searchidx, axis=0), dtype=np.float64, axis=0)
    return A

# Finding the Initial Transmission Map
def get_initial_transmission(A, brightch):
    A_c = np.max(A)
    init_t = (brightch-A_c)/(1.-A_c) # finding initial transmission map
    return (init_t - np.min(init_t))/(np.max(init_t) - np.min(init_t)) # normalized initial transmission map

# Using Dark Channel to Estimate Corrected Transmission Map
def get_corrected_transmission(I, A, darkch, brightch, init_t, alpha, omega, w):
    im = np.empty(I.shape, I.dtype);
    for ind in range(0, 3):
        im[:, :, ind] = I[:, :, ind] / A[ind] #divide pixel values by atmospheric light
    dark_c, _ = get_illumination_channel(im, w) # dark channel transmission map
    dark_t = 1 - omega*dark_c # corrected dark transmission map
    corrected_t = init_t # initializing corrected transmission map with initial transmission map
    diffch = brightch - darkch # difference between transmission maps
 
    for i in range(diffch.shape[0]):
        for j in range(diffch.shape[1]):
            if(diffch[i, j] < alpha):
                corrected_t[i, j] = dark_t[i, j] * init_t[i, j]
 
    return np.abs(corrected_t)

#Calculating the Resultant Image
def get_final_image(I, A, refined_t, tmin):
    refined_t_broadcasted = np.broadcast_to(refined_t[:, :, None], (refined_t.shape[0], refined_t.shape[1], 3)) # duplicating the channel of 2D refined map to 3 channels
    J = (I-A) / (np.where(refined_t_broadcasted < tmin, tmin, refined_t_broadcasted)) + A # finding result 
 
    return (J - np.min(J))/(np.max(J) - np.min(J)) # normalized image

def enhance_details (img, sigma_s = 10, sigma_r = 0.15):
    img = cv2.detailEnhance(img, sigma_s, sigma_r)
    return img

def enhance_edges (img, sigma_s = 64, sigma_r = 0.2, flags= 1):
    img = cv2.edgePreservingFilter(img, flags, sigma_s, sigma_r)
    return img

