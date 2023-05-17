cd .\src\
py .\train.py 2 OCR both
cd .\models
ren Trained_RF.joblib Trained_OCR_RF.joblib
ren Trained_SVM.joblib Trained_OCR_SVM.joblib
cd ..
cd .\src\
py .\train.py yasmine1 hog both
cd .\models
ren Trained_RF.joblib Trained_MASK_RF.joblib
ren Trained_SVM.joblib Trained_MASK_SVM.joblib
cd ..