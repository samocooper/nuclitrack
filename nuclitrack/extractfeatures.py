import numpy as np
from skimage.measure import regionprops
from skimage.external import tifffile


def framefeatures(files, labels, counter):
    labels = labels.astype(int)

    features_temp = []

    for j in range(len(files)):
        im = tifffile.imread(files[j])
        im = im.astype(float)
        features_temp.append(regionprops(labels, im))

    features = dict()
    features['tracking'] = np.zeros((len(features_temp[0]), 15))
    features['data'] = np.zeros((len(features_temp[0]), 13))

    dims = labels.shape
    new_label = np.zeros((dims[0], dims[1]))

    for j in range(len(features_temp[0])):

        # Tracking Features

        cell_temp = features_temp[0][j]
        ypos = cell_temp.centroid[0]
        xpos = cell_temp.centroid[1]

        features['tracking'][j, 0] = counter
        features['tracking'][j, 2] = xpos
        features['tracking'][j, 3] = ypos
        features['tracking'][j, 4] = min([ypos, dims[0] - ypos, xpos, dims[1] - xpos])

        # Morphology Features

        features['data'][j, 0] = cell_temp.area
        features['data'][j, 1] = cell_temp.eccentricity
        features['data'][j, 2] = cell_temp.major_axis_length
        features['data'][j, 3] = cell_temp.perimeter

        # Intensity Measurements for classification

        for k in range(0, len(files)):

            cell_temp = features_temp[k][j]
            mu = cell_temp.mean_intensity
            im_temp = cell_temp.intensity_image.flatten()
            bin_temp = cell_temp.image.flatten()
            im_temp = im_temp[bin_temp]

            ind = k*3 + 4
            features['data'][j, ind] = mu
            features['data'][j, ind+1] = np.std(im_temp)
            features['data'][j, ind+2] = np.std(im_temp[im_temp > mu])

        new_label[labels == cell_temp.label] = counter

        counter += 1

    return features, new_label, counter
