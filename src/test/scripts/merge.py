import os
import shutil

# Function to create new folder if not exists


def createFolder(folder_name, parent_folder_path):

    # Path
    path = os.path.join(parent_folder_path, folder_name)

    # Create the folder
    # 'new_folder' in
    # parent_folder
    try:

        # mode of the folder
        mode = 0o777

        # Create folder
        os.mkdir(path, mode)

    except OSError as error:
        print(error)


def main():
    # folder path
    current_dir = os.path.dirname(os.path.realpath(__file__))
    final_dir = current_dir + '/../../test/'
    dir_path_men = final_dir + 'predata/men/test/'
    dir_path_women = final_dir + 'predata/women/test/'

    # list of folders to be merged
    merge_dir = [dir_path_men + str(0), dir_path_women + str(0),
                 dir_path_men + str(1), dir_path_women + str(1),
                 dir_path_men + str(2), dir_path_women + str(2),
                 dir_path_men + str(3), dir_path_women + str(3),
                 dir_path_men + str(4), dir_path_women + str(4),
                 dir_path_men + str(5), dir_path_women + str(5)]

    # folder in which all the content
    # will be merged
    merge_folder = "data"

    # merge_folder path = current_folder + merge_folder
    merge_folder_path = os.path.join(final_dir, merge_folder)

    # delete
    shutil.rmtree(final_dir+merge_folder, ignore_errors=False, onerror=None)

    # create merge_folder if not exists
    createFolder(merge_folder, final_dir)

    # enumerate on list_dir to get the
    # content of all the folders ans store it in a dictionary
    content_list = {}
    for index, val in enumerate(merge_dir):
        path = os.path.join(current_dir, val)
        content_list[merge_dir[index]] = os.listdir(path)

    # loop through the list of folders
    for sub_dir in content_list:

        # loop through the contents of the
        # list of folders
        for contents in content_list[sub_dir]:

            # make the path of the content to move
            path_to_content = sub_dir + "/" + contents

            # make the path with the current folder
            dir_to_move = os.path.join(current_dir, path_to_content)

            # move the file
            shutil.move(dir_to_move, merge_folder_path)


if __name__ == '__main__':
    main()
