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

def batchanalyse(text_file, param_file, output_file, ring_flag=False):

    print('Loading Images')
    fov = h5py.File(output_file + '.hdf5', "a")
    file_list, label_files = loadimages.filelistfromtext(text_file)

    loadimages.savefilelist(file_list, fov)
    dims, min_vals, max_vals = loadimages.loadimages(file_list)
    channel_num = len(file_list)
    params = h5py.File(param_file, "a")
    s_params = params['seg_param'][...]
    frames = len(file_list[0])
    seg_ch = int(s_params[10])

    if len(label_files) > 1:
        labels = loadimages.loadlabels(label_files)

    else:
        print('Segmenting Cells')
        cpu_count = multiprocessing.cpu_count()
        pool = Pool(cpu_count)
        labels_list = pool.map(partial(segment_image, s_params, min_vals[seg_ch],
                                  max_vals[seg_ch]), file_list[seg_ch])
        pool.close()
        pool.join()

        labels = np.zeros((frames, labels_list[0].shape[0], labels_list[0].shape[1]))

        for i in range(len(labels_list)):
            labels[i, :, :] = labels_list[i]

    print('Extracting features')
    features = dict()
    features['tracking'] = np.zeros((1, 13))
    features['data'] = np.zeros((1, 22))
    counter = 1

    for i in range(frames):

        files = []
        for j in range(channel_num):
            files.append(file_list[j][i])

        temp_feat, labels[i, :, :], counter = extractfeatures.framefeatures(files, labels[i, :, :], counter, ring_flag)
        temp_feat['tracking'][:, 1] = i
        features['tracking'] = np.vstack((features['tracking'], temp_feat['tracking']))
        features['data'] = np.vstack((features['data'], temp_feat['data']))

    inds = np.argsort(features['tracking'][:, 0])
    features['tracking'] = features['tracking'][inds, :]
    features['data'] = features['data'][inds, :]
    features['tracking'][1:, 5] = 1.

    features = classifycells.classifycells(features, params['training'])
    tracking_object = trackcells.TrackCells(features=features['tracking'][...],
                                            track_param=params['track_param'][...], frames=frames)

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

    print('saving data')

    tracks, features['tracking'][...], count, double = tracking_object.get()
    tracks_stored = np.zeros(int(max(tracks[:, 4])))

    features_hdf5 = fov.create_group('features')
    features_hdf5.create_dataset("tracking", data=features['tracking'])
    features_hdf5.create_dataset("data", data=features['data'])
    fov.create_dataset("labels", data=labels)
    fov.create_dataset("tracks", data=tracks)
    fov.create_dataset("tracks_stored", data=tracks_stored)

    trackcells.save_csv(features, tracks, output_file + '.csv')

    print('done')

