# Generate the expected file .txt to be used for later comparison
import os

def main():

    # folder path
    current_dir=os.path.dirname(os.path.realpath(__file__))
    final_dir = current_dir + '/../test/'
    dir_path_men = final_dir + 'predata/men/test/'
    dir_path_women = final_dir + 'predata/women/test/'

    expected = open(final_dir + 'expected.txt',"w+")

    # Iterate directory
    for j in [0, 1, 2, 3, 4, 5]:
        print(j)

        dir_path_men = final_dir + 'predata/men/test/'
        dir_path_women = final_dir + 'predata/women/test/'

        dir_path_men = dir_path_men + str(j) + '/'
        dir_path_women = dir_path_women + str(j) + '/'

        for path in os.listdir(dir_path_men):

            # check if current path is a file
            if os.path.isfile(os.path.join(dir_path_men, path)):
                expected.write(str(j)+'\n')

        for path in os.listdir(dir_path_women):

            # check if current path is a file
            if os.path.isfile(os.path.join(dir_path_women, path)):
                expected.write(str(j)+'\n')

    expected.close()

if __name__ == '__main__':
    main()