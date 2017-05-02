import h5py
import argparse
import numpy as np
import multiprocessing
from multiprocessing import Pool
from functools import partial

from nuclitrack import loadimages
from nuclitrack import extractfeatures
from nuclitrack import classifycells
from nuclitrack import segmentimages
from nuclitrack import trackcells

parser = argparse.ArgumentParser(description='Batch Analysis')
parser.add_argument('--txtfile', dest = 'text_file')
parser.add_argument('--paramfile', dest='param_file')
parser.add_argument('--outputfile', dest='output_file')

args = parser.parse_args()

def batch_analyse(text_file=None, param_file=None, output_file=None):

    print('Loading Images')
    fov = h5py.File(output_file + '.hdf5', "a")

    file_list = loadimages.filelistfromtext(text_file)
    loadimages.savefilelist(file_list, fov)

    images = loadimages.loadimages(file_list)
    channel_num = len(images)
    params = h5py.File(param_file, "a")
    s_params = params['seg_param'][...]
    frames = images[0].shape[0]

    print('Segmenting Cells')
    cpu_count = multiprocessing.cpu_count()
    pool = Pool(cpu_count)
    labels_list = pool.map(partial(segmentimages.segment_image, s_params), [images[0][frame, :, :] for frame in range(frames)])
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

    tracking_object = trackcells.TrackCells(features, frames, np.asarray([0.05, 50, 1, 5, 0, 1, 3]))

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

batch_analyse(text_file=args.text_file, param_file=args.param_file, output_file=args.output_file)




