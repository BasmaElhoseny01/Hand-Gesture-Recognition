import os
import sys
# This is the main pipeline for testing our model

# Step(0) Import the required utilities
current_dir=os.path.abspath('')
print("Current Directory: ",current_dir)
src_dir=os.path.join(current_dir,'../')
sys.path.insert(1, src_dir) # back to the src directory

from utils import *
from global_ import IMG_SIZE
from modules.preprocessing import preprocess_img
from modules.feature_extraction import extract_features_img
import time
def main(argv):
    preprocessing_option=argv[0]
    feature_extractor_option=argv[1]
    model_option=argv[2]

    final_results = []
    time_vector = []
    
    #Step-0 Load model
    print('Loading Model ....')
    if(model_option=="svm"):
        filename = "./Trained_SVM.joblib"
    elif(model_option=="rf"):  
        filename = "./Trained_RF.joblib"
    else:
        print("Wrong Model Option!!!",model_option)
        raise TypeError("Wrong Feature Option")
    model = joblib.load(filename)
    if(model is None):
        print('Model is None')
    

    path_data_folder = './data'
    for filename in os.listdir(path_data_folder):
        try:
            # Get path
            path = path_data_folder + "/" + filename
            # Reading Image
            img = cv2.imread(path)  # YALAHWII
            if img is None:
                continue

            start=time.time()

            #Resize Image
            img = cv2.resize(img, IMG_SIZE)

            #Step-1 Preprocessing
            img = preprocess_img(img, preprocessing_option, debug=False)

            #Step-2 Extract features
            fv = extract_features_img(img,feature_extractor_option)

            #Step-3 Predict
            result = model.predict([fv])
            end = time.time()


            final_results.append(result[0])
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