import h5py
import numpy as np
import multiprocessing
from multiprocessing import Pool
from functools import partial

from . import loadimages
from . import extractfeatures
from . import classifycells
from . import trackcells
from .segmentimages import segment_image

def batchanalyse(text_file, param_file, output_file):

    print('Loading Images')
    fov = h5py.File(output_file + '.hdf5', "a")

    file_list, label_files = loadimages.filelistfromtext(text_file)

    loadimages.savefilelist(file_list, fov)
    images = loadimages.loadimages(file_list)
    channel_num = len(images)
    params = h5py.File(param_file, "a")
    s_params = params['seg_param'][...]
    frames = images[0].shape[0]

    if len(label_files) > 1:
        labels = loadimages.loadimages([label_files], label_flag=True)

    else:
        print('Segmenting Cells')
        cpu_count = multiprocessing.cpu_count()
        pool = Pool(cpu_count)
        labels_list = pool.map(partial(segment_image, s_params), [images[int(s_params[10])][frame, :, :] for frame in range(frames)])
        pool.close()
        pool.join()

        labels = np.zeros((frames, labels_list[0].shape[0], labels_list[0].shape[1]))

        for i in range(len(labels_list)):
            labels[i, :, :] = labels_list[i]

    print('Extracting features')
    feature_num = 20 + 3 * (channel_num - 1)

    counter = 1
    features = np.zeros((1, feature_num))

    for i in range(frames):

        feature_images = [labels[i, :, :]]
        for j in range(channel_num):
            feature_images.append(images[j][i, :, :])

        feature_mat, labels[i, :, :], counter = extractfeatures.framefeatures(feature_images, feature_num, counter)
        feature_mat[:, 1] = i
        features = np.vstack((features, feature_mat))

    features[1:, 17:19] = 1
    features = features[np.argsort(features[:, 0]), :]

    features = classifycells.classifycells(features, params['training_data'][...])

    tracking_object = trackcells.TrackCells(features, frames, params['track_param'][...])

    print('Tracking cells')
    counter = 0

    while tracking_object.addtrack():
        counter += 1
        print(counter)

    print('Optimising tracks')
    for i in range(2):
        counter = 0
        while tracking_object.optimisetrack():
            counter += 1
            print(counter)

    tracks, features, count, double = tracking_object.get()
    tracks_stored = np.zeros(int(max(tracks[:, 4])))

    fov.create_dataset("labels", data=labels)
    fov.create_dataset("features", data=features)
    fov.create_dataset("tracks", data=tracks)
    fov.create_dataset("tracks_stored", data=tracks_stored)

    trackcells.save_csv(features, tracks, output_file + '.csv')


