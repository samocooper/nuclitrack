from sklearn.neural_network import MLPClassifier
import numpy as np

def ellipse_roi(roi_vec, imshape):

    x_values = np.arange(roi_vec[1] - roi_vec[3], roi_vec[1] + roi_vec[3])
    y_values = np.arange(roi_vec[0] - roi_vec[2], roi_vec[0] + roi_vec[2])

    xx, yy = np.meshgrid(x_values, y_values)

    xx = xx.flatten()
    yy = yy.flatten()

    mask = np.logical_and(np.logical_and(xx > 0, xx < imshape[0]), np.logical_and(yy > 0, yy < imshape[1]))

    xx = xx[mask]
    yy = yy[mask]

    for j in range(xx.shape[0]):
        if (xx[j] - roi_vec[1]) ** 2 / roi_vec[3] ** 2 + (yy[j] - roi_vec[0]) ** 2 / roi_vec[2] ** 2 > 1:
            xx[j] = -1
            yy[j] = -1

    xx = xx[np.logical_not(xx == -1)]
    yy = yy[np.logical_not(yy == -1)]

    pixls = np.zeros((xx.shape[0], 2))
    pixls[:, 0] = xx
    pixls[:, 1] = yy

    return pixls

def unique_pixls(pxls):

    # Identify unique combinations of pixels in the array

    pxl_mask = pxls[:, 0] * 10 ** 5 + pxls[:, 1]
    pxl_mask = np.unique(pxl_mask)
    pxl_filtered = np.zeros((pxl_mask.shape[0], 2))
    pxl_filtered[:, 0] = pxl_mask // 10 ** 5
    pxl_filtered[:, 1] = pxl_mask % 10 ** 5
    pxl_filtered = pxl_filtered.astype(int)

    return pxl_filtered

def train_clf(training):

    clf = MLPClassifier(solver='lbfgs', activation='relu', alpha=1e-5,
                        random_state=1, hidden_layer_sizes=(20,), verbose=False)

    X = training['X'][...]
    y = training['y'][...]

    clf.fit(X, y)

    return clf