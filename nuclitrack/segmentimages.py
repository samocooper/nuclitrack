import ctoolsegmentation
import numpy as np

from skimage import filters
from skimage import morphology
from skimage.feature import peak_local_max
from scipy import ndimage
from PIL import Image

def clipping(im, val):

    im_temp = im.copy()

    if val != 0:
        im_temp[im > val] = val

    return im_temp


def background(im, val):

    im_temp = im.copy()

    if val != 0:
        im_temp = im_temp - ctoolsegmentation.fast_blur(im_temp, val)

    return im_temp

def blur(im, val):

    if val != 0:

        if val <= 5:
            im = filters.gaussian(im, val)

        else:

            im = filters.gaussian(im, (val / 2))
            im = filters.gaussian(im, (val / 2))
            im = filters.gaussian(im, (val / 2))

    im -= np.min(im.flatten())
    im /= np.max(im.flatten())

    return im


def threshold(im, val):

    im_bin = im > val

    return im_bin


def object_filter(im_bin, val):

    im_bin = morphology.remove_small_objects(im_bin, val)

    return im_bin


def cell_centers(im, im_bin, val):

    d_mat = ndimage.distance_transform_edt(im_bin)
    d_mat /= np.max(d_mat.flatten())

    im_cent = (1 - val) * im + val * d_mat
    im_cent[np.logical_not(im_bin)] = 0

    return [im_cent, d_mat]


def fg_markers(im_cent, im_bin, val, edges):

    local_maxi = peak_local_max(im_cent, indices=False, min_distance=int(val), labels=im_bin, exclude_border=int(edges))
    markers = ndimage.label(local_maxi)[0]
    markers[local_maxi] += 1
    k = morphology.octagon(2, 2)
    markers = morphology.dilation(markers, selem=k)

    return markers


def sobel_edges(im, val):

    if val != 0:
        if val <= 5:
            im = filters.gaussian(im, val)

        else:

            im = filters.gaussian(im, (val / 2))
            im = filters.gaussian(im, (val / 2))
            im = filters.gaussian(im, (val / 2))

    im = filters.sobel(im) + 1
    im /= np.max(im.flatten())

    return im


def watershed(markers, im_bin, im_edge, d_mat, val, edges):

    k = morphology.octagon(2, 2)

    im_bin = morphology.binary_dilation(im_bin, selem=k)
    im_bin = morphology.binary_dilation(im_bin, selem=k)
    im_bin = morphology.binary_dilation(im_bin, selem=k)

    markers_temp = markers + np.logical_not(im_bin)
    shed_im = (1 - val) * im_edge - val * d_mat

    labels = morphology.watershed(image=shed_im, markers=markers_temp)
    labels -= 1
    if edges == 1:
        edge_vec = np.hstack((labels[:, 0].flatten(), labels[:, -1].flatten(), labels[0, :].flatten(),
                              labels[-1, :].flatten()))
        edge_val = np.unique(edge_vec)
        for val in edge_val:
            if not val == 0:
                labels[labels == val] = 0

    return labels


def segment_image(params, min_val, max_val, file_name):

    im = np.asarray(Image.open(file_name))
    im = im.astype(float)
    im -= min_val
    im /= max_val

    image = clipping(im, params[0])
    image2 = background(image, params[1])
    image3 = blur(image2, params[2])
    im_bin = threshold(image3, params[3])
    im_bin = object_filter(im_bin, params[4])
    [cell_center, d_mat] = cell_centers(image3, im_bin, params[5])
    markers = fg_markers(cell_center, im_bin, params[6], params[9])
    im_edge = sobel_edges(image, params[7])
    return watershed(markers, im_bin, im_edge, d_mat, params[8], params[9])
