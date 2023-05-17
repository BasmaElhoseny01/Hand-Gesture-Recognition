import sys
import os
# caution: path[0] is reserved for script path (or '' in REPL)
current_dir=os.path.abspath('')
print("Current Directory: ",current_dir)
src_dir=os.path.join(current_dir,'./src')
sys.path.insert(1, src_dir) # back to the src directory

# All files in src are now seen here
from utils import *
from modules.preprocessing import preprocessing_yasmine

count_indx=20

for j in ['test']:
    for i in range(0, 6):
        # print(i)
        path=os.path.join(src_dir,'../data_split_resize/men/'+j+'/'+str(i)+'/')
        print(path)
        indx=0
        for filename in os.listdir(path):
            # if(indx>=count_indx):
            #     continue
            print(filename)
            indx+=1

            img = cv2.imread(path+str(filename))
            if img is None:
                continue

            #Preprocessing
            # result=preprocessing_basma(img,debug=True)
            # Path is to sav e the results from show_images
            # yarab(img,debug=True,path=os.path.join(src_dir,'../preprocessing_results/'+filename))
            preprocessing_yasmine(img, debug=True,path=os.path.join(src_dir,'../preprocessing_results/'+filename))
            #Save Results
            # cv2.imwrite(os.path.join(path_result,str(filename)),result)


        # path=os.path.join(src_dir,'../data_split_resize/women/'+j+'/'+str(i)+'/')
        # print(path)
        # indx=0
        # for filename in os.listdir(path):
        #     if(indx>=count_indx):
        #         continue
        #     print(filename)
        #     indx+=1
        #     img = cv2.imread(path+str(filename))
        #     if img is None:
        #         continue

        #     #Preprocessing
        #     # result=preprocessing_basma(img,debug=True)
        #     # Path is to sav e the results from show_images
        #     # yarab(img,debug=True,path=os.path.join(src_dir,'../preprocessing_results/'+filename))

        #     preprocessing_yasmine(img, debug=False,path=os.path.join(src_dir,'../preprocessing_results/'+filename))

        #     #Save Results
        #     # cv2.imwrite(os.path.join(path_result,str(filename)),result)