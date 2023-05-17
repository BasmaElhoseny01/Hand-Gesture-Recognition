# This is the main pipeline for training our model

# Step(0) Import the required utilities
from utils import *
from modules.preprocessing import preprocessing
from modules.feature_extraction import extract_features
from modules.training import train_model

def main(argv):
    preprocessing_option=argv[0]
    feature_extractor_option=argv[1]
    model_option=argv[2]
    
    # Step(1) Read Images
    print("Loading Files ...")
    training_path = '../data_split_resize/'
    train_images = read_images_train(training_path)
    print("Files loaded")



    # Step(2) Preprocess the images
    print("Preprocessing ...")
    OCR,classification,train_images = preprocessing(train_images,option=preprocessing_option)
    print('Preprocessing Done')



    # Step(3) Extract features
    print("Extracting Features ...")
    if(feature_extractor_option=="OCR"):  #IF OCR ALready OCR is Done
        X_train=OCR
        Y_train=classification
    elif(feature_extractor_option=="orb"):
        X_train,Y_train,visual_words=extract_features(train_images,feature_extractor_option,train=True,visual_words=None,clusters=10)

        #Save visual_words in .py file
        print("Saving Visual Words...")
        np.save('../models/visual_words.npy',visual_words)

    else:
        X_train,Y_train=extract_features(train_images,feature_extractor_option,train=True)
    print('Features Extracted')
    train_images = None

    # Step(4) Train our classifier
    print('Training Model ....')
    if(model_option=="both" or model_option=="svm"):
        start = time.time()
        svm=train_model(X_train,Y_train,option="svm")
        end = time.time()

        # Step(5) Save our model
        filename = "../models/Trained_SVM.joblib"
        joblib.dump(svm, filename)
        print('SWM Model exported')
        print('Training time = ' + str(end-start))

    if(model_option=="both" or model_option=="rf"):
        start = time.time()
        rf=train_model(X_train,Y_train,option="rf")
        end = time.time()

        # Step(5) Save our model
        filename = "../models/Trained_RF.joblib"
        joblib.dump(rf, filename)
        print('RF Model exported')
        print('Training time = ' + str(end-start))   
    if(model_option!="both" and model_option!="rf" and model_option!="svm"):
        raise TypeError("Wrong Model Option")

    print('Done Training :D ')


if __name__ == "__main__":
    main(sys.argv[1:])
