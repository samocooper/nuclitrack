import numpy as np
from skimage.measure import regionprops


def framefeatures(feature_images, feature_num, counter):

    features_temp = []
    for j in range(len(feature_images)-1):
        features_temp.append(regionprops(feature_images[0], feature_images[j + 1]))

    feature_mat = np.zeros((len(features_temp[0]), feature_num))
    dims = feature_images[0].shape
    new_label = np.zeros((dims[0], dims[1]))

    for j in range(len(features_temp[0])):

        cell_temp = features_temp[0][j]
        features_vector = np.zeros(feature_num)
        ypos = cell_temp.centroid[0]
        xpos = cell_temp.centroid[1]

        features_vector[0] = counter
        features_vector[2] = xpos
        features_vector[3] = ypos
        features_vector[4] = min([ypos, dims[0] - ypos, xpos, dims[1] - xpos])

        features_vector[5] = cell_temp.area
        features_vector[6] = cell_temp.eccentricity
        features_vector[7] = cell_temp.major_axis_length
        features_vector[8] = cell_temp.perimeter

        mu = cell_temp.mean_intensity
        im_temp = cell_temp.intensity_image.flatten()
        bin_temp = cell_temp.image.flatten()
        im_temp = im_temp[bin_temp]

        features_vector[9] = mu
        features_vector[10] = np.std(im_temp)
        features_vector[11] = np.std(im_temp[im_temp > mu])

        for k in range(1, len(feature_images)-1):

            cell_temp = features_temp[k][j]
            mu = cell_temp.mean_intensity
            im_temp = cell_temp.intensity_image.flatten()
            bin_temp = cell_temp.image.flatten()
            im_temp = im_temp[bin_temp]

            features_vector[20 + (k - 1) * 3 + 0] = mu
            features_vector[20 + (k - 1) * 3 + 1] = np.std(im_temp)
            features_vector[20 + (k - 1) * 3 + 2] = np.std(im_temp[im_temp > mu])

        feature_mat[j, :] = features_vector
        new_label[feature_images[0] == cell_temp.label] = counter

        counter += 1

    return feature_mat, new_label, counter
