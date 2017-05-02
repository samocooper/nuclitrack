from skimage.external import tifffile
import numpy as np
import os
import platform


def loadimages(file_list):

    # Function to load images for one movie from list of channels/file names

    frames = len(file_list[0])

    im_test = tifffile.imread(file_list[0][0])
    dims = im_test.shape

    ims = []

    for channel in file_list:

        channel_ims = np.zeros((frames, dims[0], dims[1]))

        for i in range(len(channel)):
            im = tifffile.imread(channel[i])
            im = im.astype(float)
            channel_ims[i, :, :] = im

        ims.append(channel_ims)

    # Standardise images such that intensities lie between 0 and 1

    for i in range(len(ims)):
        ims[i] /= np.max(ims[i].flatten())

    return ims


def savefilelist(file_list, fov):

    # Save file list in hdf5 format, requires conversion to bytes and numpy array

    file_list_bin = []
    for channel in file_list:
        file_list_bin.append([bytes(file, encoding='utf8') for file in channel])

    file_list_np = np.asarray(file_list_bin)
    fov.create_dataset('file_list', data=file_list_np)


def loadfilelist(fov):

    # Load file list from hdf5 format, requires conversion from numpy array and bytes

    file_list = []
    file_list_np = fov['file_list'][...]
    for i in range(file_list_np.shape[0]):
        file_list.append([str(file, encoding='utf8') for file in file_list_np[i, :]])

    return file_list


def filelistfromtext(text_file, full_name=False):

    if not full_name:

        # Load file list where the text file is in the same directory as the image files

        if os.path.isfile(text_file):

            if platform.system() == 'Windows':
                file_name_split = text_file.split('\\')
                file_name_split = [s + '\\' for s in file_name_split]
            else:
                file_name_split = text_file.split('/')
                file_name_split = [s + '/' for s in file_name_split]

            dir_name = ''.join(file_name_split[:-1])
            file_list = []

            with open(text_file) as f:
                for line in f:

                    if line[-1] == '\n':
                        line = line[:-1]
                    file_list.append(dir_name + line)
            return [file_list]

    else:

        # Load file list where file path is also given

        if os.path.isfile(text_file):

            file_list = []

            with open(text_file) as f:
                for line in f:

                    if line[-1] == '\n':
                        line = line[:-1]

                    file_list.append(line)

            return [file_list]


def filelistfromdir(file_name):

    # Record all file names within a directory

    if platform.system() == 'Windows':
        file_name_split = file_name.split('\\')
        file_name_split = [s + '\\' for s in file_name_split]
    else:
        file_name_split = file_name.split('/')
        file_name_split = [s + '/' for s in file_name_split]

    dir_name = ''.join(file_name_split[:-1])
    dir_list = os.listdir(dir_name)

    file_name_split = file_name.split('.')
    file_type = file_name_split[1]

    # Filter out files of a different file type

    file_list = []
    for file in dir_list:
        if not (file.find(file_type) == -1):
            file_list.append(dir_name + file)

    return [file_list]


def generatetestlist(first_name, dif_loc2):
    if (dif_loc2 + 1) < len(first_name):

        test_digit = first_name[dif_loc2 + 1]
        flag = True

        if test_digit.isdigit():
            for j in np.arange(int(test_digit) + 1, int(test_digit) + 10):
                if j < 10:

                    test_digit_temp = str(j)
                    test_name = list(first_name)
                    test_name[dif_loc2 + 1] = test_digit_temp[0]
                else:
                    test_digit_temp = str(j)
                    test_name = list(first_name)
                    test_name[dif_loc2] = test_digit_temp[0]
                    test_name[dif_loc2 + 1] = test_digit_temp[1]

                if not os.path.isfile(''.join(test_name)):
                    flag = False
        else:
            flag = False

        if flag:
            dif_loc2 += 1
            dif_loc2 = generatetestlist(first_name, dif_loc2)

    return dif_loc2


def autofilelist(first_name, last_name):

    # Determine whether a difference of one exists somewhere in file names

    fnm = list(first_name)

    l1 = np.asarray([ord(i) for i in list(first_name)])
    l2 = np.asarray([ord(i) for i in list(last_name)])

    dif = l2 - l1

    dif_loc = np.where(dif)[0][0]
    dif_loc2 = np.where(dif)[0][-1]

    # Use recursion to test if last digit is in fact last digit

    dif_loc2 = generatetestlist(first_name, dif_loc2)

    # Determine range of image time points

    nm = dif[dif_loc:dif_loc2 + 1]
    mul = 10 ** len(nm)
    tot = 0
    for x in nm:
        mul //= 10
        tot += x * mul

    pad = len(str(tot))

    start_num = ''
    for i in range(dif_loc, dif_loc + pad):
        start_num += fnm[i]

    start_num = int(start_num)

    # Generate list fo file names using padded time point numbers

    file_list = []

    for i in range(tot + 1):
        s = str(start_num + i).zfill(pad)

        for j in range(pad):
            fnm[dif_loc + j] = s[j]

        file_list.append(''.join(fnm))

    return file_list
