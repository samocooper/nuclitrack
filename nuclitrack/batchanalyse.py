import multiprocessing
from functools import partial
from multiprocessing import Pool

import h5py
import numpy as np

from .nuclitrack_tools import classifycells
from .nuclitrack_tools import classifypixels
from .nuclitrack_tools import extractfeats
from .nuclitrack_tools import loadimages
from .nuclitrack_tools import movieobj
from .nuclitrack_tools import trackcells
from .nuclitrack_tools import segmentimages

def batch_analyse(text_file, param_file, output_file, parallel_flag=False, ring_flag=False):

    print('Loading Images')
    fov = h5py.File(output_file + '.hdf5', "a")
    file_list, label_files = loadimages.filelistfromtext(text_file)

    loadimages.savefilelist(file_list, fov)
    movie = movieobj.MovieObj(file_list)

    params = h5py.File(param_file, "a")
    s_params = params['seg_param'][...]
    clf = 0

    if 'seg_training' in params:
        clf = classifypixels.train_clf(params['seg_training'])

    if len(label_files) > 1:
        labels = loadimages.loadlabels(label_files)

    else:
        if parallel_flag:

            print('Segmenting Cells in Parallel')

            cpu_count = multiprocessing.cpu_count()
            pool = Pool(cpu_count)

            labels_list = pool.map(partial(segmentimages.segment_image, movie, s_params, clf), range(movie.frames))

            pool.close()
            pool.join()

            labels = np.zeros((movie.frames, movie.dims[0], movie.dims[1]))

            for i in movie.frames:
                labels[i, :, :] = labels_list[i]

        else:

            print('Segmenting Cells...')

            labels = np.zeros((movie.frames, movie.dims[0], movie.dims[1]))

            for i in range(movie.frames):

                print('Images segmented: ', i, end='\r')
                labels[i, :, :] = segmentimages.segment_image(movie, s_params, clf, i)

    print('Extracting features')

    features = dict()
    features['tracking'] = np.zeros((1, 13))
    features['data'] = np.zeros((1, 22))
    counter = 1

    for i in range(movie.frames):

        temp_feat, labels[i, :, :], counter = extractfeats.framefeats(movie, i, labels[i, :, :], counter, ring_flag)
        temp_feat['tracking'][:, 1] = i
        features['tracking'] = np.vstack((features['tracking'], temp_feat['tracking']))
        features['data'] = np.vstack((features['data'], temp_feat['data']))

    inds = np.argsort(features['tracking'][:, 0])
    features['tracking'] = features['tracking'][inds, :]
    features['data'] = features['data'][inds, :]
    features['tracking'][1:, 5] = 1.

    features = classifycells.classifycells(features, params['training'])

    tracking_object = trackcells.TrackCells(features=features['tracking'][...],
                                            track_param=params['track_param'][...], frames=movie.frames)

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

    print('Saving data')

    tracks, features['tracking'][...], count, double = tracking_object.get()
    tracks_stored = np.zeros(int(max(tracks[:, 4])))

    features_hdf5 = fov.create_group('features')
    features_hdf5.create_dataset("tracking", data=features['tracking'])
    features_hdf5.create_dataset("data", data=features['data'])
    fov.create_dataset("labels", data=labels)
    fov.create_dataset("tracks", data=tracks)
    fov.create_dataset("tracks_stored", data=tracks_stored)

    trackcells.save_csv(features, tracks, output_file + '.csv')

    print('Finished')

