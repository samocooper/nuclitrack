import numpy as np
cimport numpy as np
cimport cython

DTYPE1 = np.float
ctypedef np.float_t DTYPE1_t

DTYPE2 = np.int
ctypedef np.int_t DTYPE2_t

@cython.boundscheck(False)
@cython.wraparound(False)

def scale_im(np.ndarray[DTYPE1_t, ndim=1] data, int scale):

    cdef int wid = data.size

    cdef float mx = data[0]
    cdef float mn = data[0]

    cdef int x

    for x in range(wid):

        if data[x] > mx:
            mx = data[x]

        if data[x] < mn:
            mn = data[x]

    cdef float sc

    if mx != mn:
        sc = scale/(mx-mn)
    else:
        sc = scale

    cdef np.ndarray[DTYPE1_t, ndim=1] scaled = np.zeros([wid], dtype=DTYPE1)

    for x in range(wid):

        scaled[x] = (data[x]-mn)*sc;

    return scaled

def mat_to_im(np.ndarray[DTYPE2_t, ndim=1] data, np.ndarray[DTYPE2_t, ndim=1] cmap):

    cdef int wid = data.size
    cdef int x, x1, y

    cdef np.ndarray[DTYPE2_t, ndim=1] im = np.zeros([wid*3], dtype=DTYPE2)

    for x in range(wid):

        x1 = x*3
        y = data[x]*3

        im[x1] = cmap[y]
        im[x1+1] = cmap[y+1]
        im[x1+2] = cmap[y+2]

    return im

def indexed_mat_to_im(np.ndarray[DTYPE2_t, ndim=1] data, np.ndarray[DTYPE2_t, ndim=1] cmap, np.ndarray[DTYPE2_t, ndim=1] mapping):

    cdef int wid = data.size
    cdef int x, x1, y, ind

    cdef np.ndarray[DTYPE2_t, ndim=1] im = np.zeros([wid*3], dtype=DTYPE2)

    for x in range(wid):

        x1 = x*3
        ind = data[x]

        y = mapping[ind]*3

        im[x1] = cmap[y]
        im[x1+1] = cmap[y+1]
        im[x1+2] = cmap[y+2]

    return im