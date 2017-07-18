import numpy as np
from skimage.measure import regionprops
from skimage.morphology import dilation
from skimage.morphology import square
from PIL import Image
def framefeatures(files, labels, counter, ring_flag):

    labels = labels.astype(int)
    labels_bin = labels == 0
    features_temp = []
    ims = []

    for j in range(len(files)):
        im = np.asarray(Image.open(files[j]))
        im = im.astype(float)
        features_temp.append(regionprops(labels, im))
        if ring_flag:
            ims.append(im)

    features = dict()
    features['tracking'] = np.zeros((len(features_temp[0]), 13))
    features['data'] = np.zeros((len(features_temp[0]), 22))

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

        # Determine region for dilation

        if ring_flag:
            r = int(np.round(cell_temp.perimeter/5))

            bbox = cell_temp.bbox
            bbox_dil = (np.maximum(0, bbox[0]-r), np.maximum(0, bbox[1]-r),
                        np.minimum(dims[0], bbox[2]+r-1), np.minimum(dims[1], bbox[3]+r-1))
            pad = ((bbox[0]-bbox_dil[0], bbox_dil[2]-bbox[2]), (bbox[1] - bbox_dil[1], bbox_dil[3]-bbox[3]))

            image_dil = np.pad(cell_temp.image, pad, 'constant')
            image_dil = dilation(image_dil, square(r))

            bin_roi = labels_bin[bbox_dil[0]:bbox_dil[2], bbox_dil[1]:bbox_dil[3]]
            ring_region = np.logical_and(image_dil, bin_roi)

        # Intensity Measurements for classification

        for k in range(0, len(files)):

            cell_temp = features_temp[k][j]
            mu = cell_temp.mean_intensity
            im_temp = cell_temp.intensity_image.flatten()
            bin_temp = cell_temp.image.flatten()
            im_temp = im_temp[bin_temp]

            ind = k*6 + 4
            features['data'][j, ind] = mu
            features['data'][j, ind+1] = np.median(im_temp)
            features['data'][j, ind+2] = np.std(im_temp)
            features['data'][j, ind+3] = np.std(im_temp[im_temp > mu])

            if ring_flag:

                im_roi = ims[k][bbox_dil[0]:bbox_dil[2], bbox_dil[1]:bbox_dil[3]]
                ring_region_vals = im_roi[ring_region].flatten()

                features['data'][j, ind+4] = np.mean(ring_region_vals)
                features['data'][j, ind+5] = np.median(ring_region_vals)

        new_label[labels == cell_temp.label] = counter
        counter += 1

    return features, new_label, counter
