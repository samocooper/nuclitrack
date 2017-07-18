from PIL import Image
import numpy as np
import os

def loadimages(file_list):

    # Function to load images for one movie from list of channels/file names

    frames = len(file_list[0])
    im_test = np.asarray(Image.open(file_list[0][0]))
    dims = im_test.shape

    min_vals = []
    max_vals = []

    for j in range(len(file_list)):

        im_test = np.asarray(Image.open(file_list[0][0]))
        max_val = np.max(im_test)
        min_val = np.min(im_test)

        for i in range(frames):

            im = np.asarray(Image.open(file_list[0][0]))
            im = im.astype(float)

            if np.max(im) > max_val:
                max_val = np.max(im)
            if np.min(im) < min_val:
                min_val = np.min(im)

        min_vals.append(min_val)
        max_vals.append(max_val)

    return dims, min_vals, max_vals

def loadlabels(file_list):

    im_test = np.asarray(Image.open(file_list[0]))
    dims = im_test.shape

    labels = np.zeros((len(file_list), dims[0], dims[1]))

    for i in range(len(file_list)):
        labels[i, :, :] = np.asarray(Image.open(file_list[i]))

    return labels


def savefilelist(file_list, fov):

    # Save file list in hdf5 format, requires conversion to bytes and numpy array

    file_list_bin = []
    for channel in file_list:
        file_list_bin.append([bytes(file, encoding='utf8') for file in channel])

    file_list_np = np.asarray(file_list_bin)

    for g in fov:
        if g =='file_list':
            del fov['file_list']

    fov.create_dataset('file_list', data=file_list_np)


def loadfilelist(fov):

    # Load file list from hdf5 format, requires conversion from numpy array and bytes

    file_list = []
    file_list_np = fov['file_list'][...]
    for i in range(file_list_np.shape[0]):
        file_list.append([str(file, encoding='utf8') for file in file_list_np[i, :]])

    return file_list


def filelistfromtext(text_file):

    # Load file list where the text file is in the same directory as the image files

    dir_name = os.path.dirname(text_file)
    file_list = []
    channel_list = []
    prev = 1
    label_list = []

    with open(text_file) as f:
        for line in f:

            line = line.strip()
            line_split = line.split(',')

            if len(line_split) > 1:

                line_split[0] = line_split[0].strip()
                line_split[1] = line_split[1].strip()

                if int(line_split[0]) == 0:
                    label_list.append(os.path.join(dir_name, line_split[1]))

                else:
                    if not int(line_split[0]) == prev:
                        prev = int(line_split[0])
                        file_list.append(channel_list)
                        channel_list = []

                    channel_list.append(os.path.join(dir_name, line_split[1]))

            else:
                channel_list.append(os.path.join(dir_name, line))

    file_list.append(channel_list)

    return file_list, label_list


def filelistfromdir(file_name):

    # Record all file names within a directory

    dir_name = os.path.dirname(file_name)
    dir_list = os.listdir(dir_name)

    file_name_split = file_name.split('.')
    file_type = file_name_split[1]

    # Filter out files of a different file type

    file_list = []
    for file in dir_list:
        if not (file.find(file_type) == -1):
            file_list.append(os.path.join(dir_name, file))
    file_list.sort()

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
