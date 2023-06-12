import os


def main():

    # folder path
    current_dir = os.path.dirname(os.path.realpath(__file__))
    final_dir = current_dir + '/../../test/'
    dir_path_men = final_dir + 'predata/men/test/'
    dir_path_women = final_dir + 'predata/women/test/'

    i = 1
    for j in [0, 1, 2, 3, 4, 5]:

        for path in os.listdir(dir_path_men + str(j) + '/'):
            os.rename(dir_path_men + str(j) + '/' + path,
                      dir_path_men + str(j) + '/' + str(i) + '.jpg')
            i += 1

        for path in os.listdir(dir_path_women + str(j) + '/'):
            os.rename(dir_path_women + str(j) + '/' + path,
                      dir_path_women + str(j) + '/' + str(i) + '.jpg')
            i += 1


if __name__ == '__main__':
    main()
