import os
import random
import shutil
import sys
import getopt
import splitfolders


def simplified_data_set(ratio=8):

    for i in range(0, 6):
        source = './data/men/'+str(i)
        destination = './data_simple/men/'+str(i)
        pick_random(source, destination, int(ratio))

    for i in range(0, 6):
        source = './data/women/'+str(i)
        destination = './data_simple/women/'+str(i)
        pick_random(source, destination, int(ratio))


def pick_random(source, destination, ratio=8):

    isExist = os.path.exists(destination)
    if not isExist:
        os.makedirs(destination)
        print("The new directory is created!")

    shutil.rmtree(destination, ignore_errors=False, onerror=None)
    os.mkdir(destination)
    onlyfiles = [f for f in os.listdir(
        source) if os.path.isfile(os.path.join(source, f))]
    no_of_files = round((len(onlyfiles)/ratio))
    for i in range(no_of_files):
        files = [filenames for (filenames) in os.listdir(source)]
        random_file = random.choice(files)
        shutil.copy(f'{source}\\{random_file}', destination)

    return None


def split_data_simple(seed=42):
    current_dir=os.path.dirname(os.path.realpath(__file__))
    input_folder = os.path.join(current_dir,'../../data_resize/men/')
    output_folder = os.path.join(current_dir,'../../data_split_resize/men/')

    isExist = os.path.exists(output_folder)
    if isExist:
        shutil.rmtree(output_folder, ignore_errors=False, onerror=None)
        os.makedirs(output_folder)
    splitfolders.ratio(input_folder, output=output_folder,
                    seed=seed, ratio=(.9, .0, .1), group_prefix=None)
    #                           (train, val, test)

    input_folder = os.path.join(current_dir,'../../data_resize/women/')
    output_folder = os.path.join(current_dir,'../../data_split_resize/women/')

    isExist = os.path.exists(output_folder)
    if isExist:
        shutil.rmtree(output_folder, ignore_errors=False, onerror=None)
        os.makedirs(output_folder)
    splitfolders.ratio(input_folder, output=output_folder,
                    seed=seed, ratio=(.9, .0, .1), group_prefix=None)
    #                           (train, val, test)
    return None


def main(argv):
    argumentList = argv

    # Options :=requires argument
    # options = "sij"
    options = "st"

    # Long options
    # long_options = ["all", "My_file", "Output="]
    long_options = ["simple", "split"]

    try:
        # Parsing argument
        arguments, values = getopt.getopt(argumentList, options, long_options)
        print(arguments)
        print(values)

        # checking for all argument
        for currentArgument, currentValue in arguments:
            # print(currentArgument)

            if currentArgument in ("-s", "--simple"):
                print("Generating Simple Data Set Ratio: ", values[0])
                simplified_data_set(values[0])
                return

            if currentArgument in ("-t", "--split"):
                print("Splitting Simple Data Set")
                split_data_simple(int(values[0]))
                return

            # if currentArgument in ("-i", "--large"):
                # i=values[0]
                print("Running Test Case "+i)
                # runTestCase(int(i))
                # return

    except getopt.error as err:
        # output error, and    return with an error code
        print(str(err))


if __name__ == '__main__':
    main(sys.argv[1:])
