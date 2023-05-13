# This is the main pipeline for testing our model

# Step(0) Import the required utilities
from utils import *
from modules.preprocessing import preprocessing
from modules.feature_extraction import extract_features

def main(argv):
    preprocessing_option=argv[0]
    feature_extractor_option=argv[1]
    model_option=argv[2]

    # Step(1) Read Images
    print("Loading Files ...")
    path='../data_split_resize/'
    test_images=read_images_test(path,type="test")
    print('Images loaded')

    # Step(2) Preprocess the images
    print("Preprocessing ...")
    OCR,classification,test_images=preprocessing(test_images,option=preprocessing_option)
    print('Preprocessing Done')

    # Step(3) Extract features
    # Note: We need to edit this part in order not to read images based on classes (i.e. one bulk of images)
    print("Extracting Features ...")
    if(feature_extractor_option=="OCR"):  #IF OCR ALready OCR is Done
        X_test=OCR
        Y_test=classification
    else:
        X_test,Y_test=extract_features(test_images,feature_extractor_option)
    print('Features Extracted')
    test_images = None


    # Step(4) Load our model SVM
    print('Loading Model ....')
    if(model_option=="both" or model_option=="svm"):
        filename = "../models/Trained_SVM.joblib"
        loaded_model = joblib.load(filename)
        print('SVM Model loaded')

        # Step(5) Evaluate the results
        result = loaded_model.predict(X_test)
        print('SVM Results evaluated')

        # Step(6) Write the results into txt file
        y = ["{}\n".format(i) for i in result]
        with open('../results/svm_results.txt', 'w') as fp:
            fp.writelines(y)
        print('SVM Results saved')


     # Step(4) Load our model SVM
    if(model_option=="both" or model_option=="rf"):  
        filename = "../models/Trained_RF.joblib"
        loaded_model = joblib.load(filename)
        print('RF Model loaded')

        # Step(5) Evaluate the results
        result = loaded_model.predict(X_test)
        print('RF Results evaluated')

        # Step(6) Write the results into txt file
        y = ["{}\n".format(i) for i in result]
        with open('../results/rf_results.txt', 'w') as fp:
            fp.writelines(y)
        print('RF Results saved')
        
    if(model_option!="both" and model_option!="rf" and model_option!="svm"):
        raise TypeError("Wrong Model Option")

    # step(7) save expected results
    y = ["{}\n".format(i) for i in Y_test]
    with open('../results/expected.txt', 'w') as fp:
        fp.writelines(y)
    print('Expected outputs saved')

    print('Done Testing :D ')

if __name__ == "__main__":
    main(sys.argv[1:])
