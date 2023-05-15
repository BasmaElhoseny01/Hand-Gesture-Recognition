# Imports
import cv2
import shutil
import os
import sys


def main(argv):
    print('Hello')
    new_size = (int(argv[0]), int(argv[1]))
    print("Resize", new_size)

    current_dir=os.path.dirname(os.path.realpath(__file__))
    for j in ['test', 'val', 'train']:
        print(j)

        for i in range(0, 6):
            print(i)
            path = os.path.join(current_dir,'../../data_simple/split/men/'+j+'/'+str(i)+'/')
            path_result = os.path.join(current_dir,'../../data_split_resize/men/'+j+'/'+str(i)+'/')

            print("Resizing images in "+path)
            print("dist", path_result)

            isExist = os.path.exists(path)

            if not isExist:
                print("The directory doesn't exist !!", path)
                break

            isExist = os.path.exists(path_result)
            if not isExist:
                os.makedirs(path_result)
                print("The new directory is created!")
            else:
                shutil.rmtree(path_result, ignore_errors=False, onerror=None)
                os.mkdir(path_result)

            for filename in os.listdir(path):
                print(filename)
                img = cv2.imread(path+str(filename))
                if img is None:
                    continue
                img = cv2.resize(img, new_size)
                cv2.imwrite(path_result+str(filename), img)

            # Women
            path = os.path.join(current_dir,'../../data_simple/split/women/'+j+'/'+str(i)+'/')
            path_result = os.path.join(current_dir,'../../data_split_resize/women/'+j+'/'+str(i)+'/')

            print("Resizing images in "+path)
            print("dist", path_result)

            isExist = os.path.exists(path)

            if not isExist:
                print("The directory doesn't exist !!", path)
                break

            isExist = os.path.exists(path_result)
            if not isExist:
                os.makedirs(path_result)
                print("The new directory is created!")
            else:
                shutil.rmtree(path_result, ignore_errors=False, onerror=None)
                os.mkdir(path_result)

            for filename in os.listdir(path):
                print(filename)
                img = cv2.imread(path+str(filename))
                if img is None:
                    continue
                img = cv2.resize(img, new_size)
                cv2.imwrite(path_result+str(filename), img)

    return


if __name__ == "__main__":
    main(sys.argv[1:])
