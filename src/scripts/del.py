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

    del_images=['0_men (1).JPG',
                '2_men (1).JPG',
                '2_men (7).JPG',
                '2_men (10).JPG',
                '2_men (22).JPG',
                '3_men (13).JPG',
                '3_men (38).JPG',
                '3_men (34).JPG',
                '3_men (88).JPG',
                '3_men (139).JPG',
                '3_men (155).JPG',
                '4_men (26).JPG',
                '4_men (27).JPG',
                '4_men (51).JPG',
                '4_men (64).JPG',
                '4_men (67).JPG',
                '4_men (84).JPG',
                '4_men (141).JPG',
                '5_men (10).JPG',
                '5_men (11).JPG',
                '5_men (12).JPG',
                '5_men (82).JPG',
                '5_men (46).JPG',
                '2_women (79).JPG',
                '3_women (13).JPG',
                '3_women (55).JPG',
                '3_women (68).JPG',
                '3_women (75).JPG',
                '3_women (83).JPG',
                '3_women (90).JPG',
                '3_women (103).JPG',
                '3_women (112).JPG',
                '4_women (79).JPG',
                '4_women (81).JPG',
                '4_women (83).JPG',
                '4_women (84).JPG',
                '5_women (81).JPG',
                '5_women (83).JPG',
                '5_women (84).JPG']
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
