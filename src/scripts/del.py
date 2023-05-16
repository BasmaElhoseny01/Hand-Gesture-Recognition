import os
import random
import shutil
import sys
import getopt
import splitfolders




def main():
    
    current_dir=os.path.dirname(os.path.realpath(__file__))
    root_dir=current_dir+'/../../data_resize'
    print("root_dir:",root_dir)

    del_images=['0_men (1).JPG']
    # del_images=[]
    print("Moving",del_images)

    dst_dir=os.path.join(root_dir,"outliers")
    isExist = os.path.exists(dst_dir)
    if not isExist:
        os.makedirs(dst_dir)
        print("The new directory is created! at",dst_dir)


    for folder,subfolder,files in os.walk(root_dir):
            # print(folder)
            if(folder.__contains__('outliers')):
                 continue
                #  print(folder)
            if(subfolder==[]):
                # print(subfolder)
                # print(os.path.join(folder)) #D:\Hand-Gesture-Recognition\src\scripts/../../data_split_resize\men\val\5
                for f in files:
                    # print(f)
                    if(f in del_images):
                        path=os.path.join(folder,f)
                        # print("path",path)
                        # print(os.path.join(dst_dir,f))
                        shutil.move(path, os.path.join(dst_dir,f))
                        print("Moved",path,"->",os.path.join(dst_dir,f))

    


if __name__ == '__main__':
    main()
