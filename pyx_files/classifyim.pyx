import numpy as np
cimport numpy as np
cimport cython

DTYPE1 = np.float
ctypedef np.float_t DTYPE1_t

DTYPE2 = np.int
ctypedef np.int_t DTYPE2_t

@cython.boundscheck(False)
@cython.wraparound(False)

def classify_im(np.ndarray[DTYPE1_t, ndim=2] im, int wsize, int stride, int im_hgt, int im_wid):

    cdef int i,j
    cdef int wid
    cdef int hgt
    cdef int count

    wid = im.shape[1]
    hgt = im.shape[0]

    cdef np.ndarray[DTYPE1_t, ndim=2] im2 = np.zeros((im_hgt*im_wid,4*(wsize//stride)*(wsize//stride)), dtype=DTYPE1)

    count = 0
    for i in range(wsize, hgt-wsize):
        for j in range(wsize, wid-wsize):
            im2[count,:] = im[i-wsize:i+wsize:stride,j-wsize:j+wsize:stride].flatten()
            count = count + 1
    return im2