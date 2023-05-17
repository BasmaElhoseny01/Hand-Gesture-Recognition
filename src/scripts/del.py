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
                '0_men (27).JPG',
                '0_men (28).JPG',
                '0_men (35).JPG',
                '0_men (106).JPG',
                '0_men (118).JPG',
                '0_men (163).JPG',
                '1_men (13).JPG',
                '1_men (33).JPG',
                '1_men (34).JPG',
                '1_men (60).JPG',
                '1_men (91).JPG',
                '1_men (94).JPG',
                '1_men (100).JPG',
                '1_men (135).JPG',
                '1_men (137).JPG',
                '1_men (139).JPG',
                '1_men (162).JPG',
                '2_men (1).JPG',
                '2_men (7).JPG',
                '2_men (9).JPG',
                '2_men (10).JPG',
                '2_men (22).JPG',
                '2_men (33).JPG',
                '2_men (34).JPG',
                '2_men (46).JPG',
                '2_men (61).JPG',
                '2_men (100).JPG',
                '2_men (142).JPG',
                '2_men (167).JPG',
                '3_men (13).JPG',
                '3_men (37).JPG',
                '3_men (38).JPG',
                '3_men (62).JPG',
                '3_men (63).JPG',
                '3_men (88).JPG',
                '3_men (99).JPG',
                '3_men (106).JPG',
                '3_men (124).JPG',
                '3_men (126).JPG',
                '3_men (139).JPG',
                '3_men (145).JPG',
                '3_men (148).JPG',
                '3_men (155).JPG',
                '4_men (1).JPG',
                '4_men (13).JPG',
                '4_men (16).JPG',
                '4_men (26).JPG',
                '4_men (27).JPG',
                '4_men (51).JPG',
                '4_men (63).JPG',
                '4_men (64).JPG',
                '4_men (67).JPG',
                '4_men (73).JPG',
                '4_men (75).JPG',
                '4_men (76).JPG',
                '4_men (77).JPG',
                '4_men (78).JPG',
                '4_men (79).JPG',
                '4_men (84).JPG',
                '4_men (94).JPG',
                '4_men (119).JPG',
                '4_men (121).JPG',
                '4_men (133).JPG',
                '4_men (141).JPG',
                '5_men (10).JPG',
                '5_men (11).JPG',
                '5_men (12).JPG',
                '5_men (82).JPG',
                '5_men (46).JPG',
                '5_men (58).JPG',
                '0_woman (12).JPG',
                '0_woman (13).JPG',
                '0_woman (19).JPG',
                '0_woman (31).JPG',
                '0_woman (71).JPG',
                '0_woman (92).JPG',
                '1_woman (8).JPG',
                '1_woman (61).JPG',
                '1_woman (71).JPG',
                '1_woman (92).JPG',
                '1_woman (95).JPG',
                '1_woman (99).JPG',
                '1_woman (105).JPG',
                '2_woman (28).JPG',
                '2_woman (40).JPG',
                '2_woman (61).JPG',
                '2_woman (71).JPG',
                '2_woman (79).JPG',
                '2_woman (87).JPG',
                '2_woman (97).JPG',
                '2_woman (116).JPG',
                '2_woman (124).JPG',
                '3_woman (4).JPG',
                '3_woman (7).JPG',
                '3_woman (11).JPG',
                '3_woman (13).JPG',
                '3_woman (31).JPG',
                '3_woman (44).JPG',
                '3_woman (50).JPG',
                '3_woman (55).JPG',
                '3_woman (68).JPG',
                '3_woman (75).JPG',
                '3_woman (83).JPG',
                '3_woman (90).JPG',
                '3_woman (98).JPG',
                '3_woman (103).JPG',
                '3_woman (107).JPG',
                '3_woman (110).JPG',
                '3_woman (112).JPG',
                '3_woman (116).JPG',
                '4_woman (1).JPG',
                '4_woman (13).JPG',
                '4_woman (71).JPG',
                '4_woman (79).JPG',
                '4_woman (81).JPG',
                '4_woman (83).JPG',
                '4_woman (84).JPG',
                '4_woman (87).JPG',
                '4_woman (95).JPG',
                '4_woman (104).JPG',
                '4_woman (110).JPG',
                '5_woman (71).JPG',
                '5_woman (81).JPG',
                '5_woman (83).JPG',
                '5_woman (84).JPG']
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
