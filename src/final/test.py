import time

import sys
import os
# Step(0) Import the required utilities
current_dir=os.path.abspath('')
print("Current Directory: ",current_dir)
src_dir=os.path.join(current_dir,'../')
sys.path.insert(1, src_dir) # back to the src directory

from modules.feature_extraction import extract_features_img
from modules.preprocessing import preprocess_img
from global_ import IMG_SIZE
from utils import *
import os
import sys
from threading import Thread

# This is the main pipeline for testing our model

# Step(0) Import the required utilities
current_dir = os.path.abspath('')
print("Current Directory: ", current_dir)
src_dir = os.path.join(current_dir, '../')
sys.path.insert(1, src_dir)  # back to the src directory




def ocr_thread(img, model, debug):
    # Step-1 Preprocessing
    img = preprocess_img(img, "2", debug=debug)

    # Step-2 Extract features
    fv = extract_features_img(img, "OCR")


    # Step-3 Predict
    result = model.predict([fv])

    return result[0]


def mask_thread(img, model, debug):

    # Step-1 Preprocessing
    img = preprocess_img(img, "yasmine1", debug=debug)
   

    # Step-2 Extract features
    fv = extract_features_img(img, "hog")


    # Step-3 Predict
    result = model.predict([fv])


    return result[0]

class ThreadWithReturnValue(Thread):
    
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}, Verbose=None):
        Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None

    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args,
                                                **self._kwargs)
    def join(self, *args):
        Thread.join(self, *args)
        return self._return
    
def main(argv):
    preprocessing_option = argv[0]
    feature_extractor_option = argv[1]
    model_option = argv[2]

    final_results = []
    time_vector = []

    # Step-0 Load models
    print('Loading Model OCR_SVM ....')
    filename = "./Trained_OCR_RF.joblib"
    ocr_svm_model = joblib.load(filename)


    print('Loading Model OCR_RF ....')
    filename = "./Trained_OCR_RF.joblib"
    ocr_rf_model = joblib.load(filename)

    print('Loading Model Mask_SVM ....')
    filename = "./Trained_MASK_SVM.joblib"
    svm_model = joblib.load(filename)

    print('Loading Model MASK_RF ....')
    filename = "./Trained_MASK_RF.joblib"
    rf_model = joblib.load(filename)

    print("Models Loaded Successfully :D")

    path_data_folder = './data'
    files = os.listdir(path_data_folder)
    files = sorted(files, key=lambda x: int(x.split('.')[0]))
    for filename in files:
        print(filename)
        fre_array=np.zeros((6,1))
        try:
            # Get path
            path = path_data_folder + "/" + filename
            # Reading Image
            img = cv2.imread(path)  # YALAHWII
            if img is None:
                continue

            start = time.time()
            # Resize Image
            img = cv2.resize(img, IMG_SIZE)

            # Create 3 Threads
            # Start a new thread to Receive messages

            # th1= ThreadWithReturnValue(target=ocr_thread, args=(img, ocr_svm_model, False))
            th2= ThreadWithReturnValue(target=ocr_thread, args=(img, ocr_rf_model, False))
            th3= ThreadWithReturnValue(target=mask_thread, args=(img, rf_model, False))
            th4= ThreadWithReturnValue(target=mask_thread, args=(img, svm_model, False))

            # th1.start()
            th2.start()
            th3.start()
            th4.start()


            # v1=th1.join()
            v2=th2.join()
            v3=th3.join()
            v4=th4.join()
            

            # fre_array[v1]+=1
            fre_array[v2]+=1
            fre_array[v3]+=1
            fre_array[v4]+=1


            if(np.max(fre_array)==1):
                result=v4
            else:
                result=np.argmax(fre_array)
            end=time.time()

            print(v2,v3,v4,result)

            final_results.append(result)
            time_vector.append(end-start)
        except Exception as e:
            print('An exception occured in stage: ', str(e))
            final_results.append(-1)
            time_vector.append(-1)

    ############
    # Step-4 Write the results into txt file
    y = ["{}\n".format(i) for i in final_results]
    with open('./results.txt', 'w') as fp:
        fp.writelines(y)
    # Write times into txt file
    y = ["{}\n".format(i) for i in time_vector]
    with open('./time.txt', 'w') as fp:
        fp.writelines(y)
    print('Results saved')


if __name__ == "__main__":
    main(sys.argv[1:])
