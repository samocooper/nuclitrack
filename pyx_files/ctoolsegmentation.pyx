from __future__ import division
import numpy as np

cimport numpy as np

DTYPE = np.float
ctypedef np.float_t DTYPE_t

cimport cython
@cython.boundscheck(False)


def fast_blur_stack(np.ndarray[DTYPE_t, ndim=3] img_stack, int r):

    cdef int f = img_stack.shape[0]
    cdef int h = img_stack.shape[1]
    cdef int w = img_stack.shape[2]
    cdef int i

    cdef np.ndarray[DTYPE_t, ndim=3] blr_stack = np.zeros((f,h,w), dtype=DTYPE)

    for i in range(f):
        blr = boxBlur(img_stack[i,:,:], w, h, r)
        blr = boxBlur(blr, w, h, r)
        blr_stack[i,:,:] = boxBlur(blr, w, h, r)

    return blr_stack

def fast_blur(np.ndarray[DTYPE_t, ndim=2] img, int r):

    cdef int h = img.shape[0]
    cdef int w = img.shape[1]

    blr = boxBlur(img, w, h, r)
    blr = boxBlur(blr, w, h, r)
    blr = boxBlur(blr, w, h, r)

    return blr

def boxBlur(np.ndarray[DTYPE_t, ndim=2] img, int w, int h, int r):

    blr = boxBlurT(img, w, h, r);
    blr = boxBlurH(blr, w, h, r);

    return blr

def boxBlurH (np.ndarray[DTYPE_t, ndim=2] img, int w, int h, int r):

    cdef np.ndarray[DTYPE_t, ndim=2] blr = np.zeros((h,w), dtype=DTYPE)
    cdef int i, j, x, x1, x2
    cdef float val
    cdef float weight = r+r+1

    for i in range(h):

        val = 0

        for x in range(0-r-1,0+r):
                if x < 0:
                    x = 0
                val += img[i,x]

        blr[i,0] = val/weight

        for j in range(w):

            x1 = j-r-1
            if x1 < 0:
                    x1 = 0
            val -= img[i,x1]

            x2 = j+r
            if x2 > w-1:
                    x2 = w-1
            val += img[i,x2]

            blr[i,j] = val/weight

    return blr

def boxBlurT (np.ndarray[DTYPE_t, ndim=2] img, int w, int h, int r):

    cdef np.ndarray[DTYPE_t, ndim=2] blr = np.zeros((h,w), dtype=DTYPE)
    cdef int i, j, y, y1, y2
    cdef float val
    cdef float weight = r+r+1

    for j in range(w):

        val = 0

        for y in range(0-r-1,0+r):
                if y < 0:
                    y = 0
                val += img[y,j]

        blr[0,j] = val/weight

        for i in range(h):

            y1 = i-r-1
            if y1 < 0:
                    y1 = 0
            val -= img[y1,j]

            y2 = i+r
            if y2 > h-1:
                    y2 = h-1
            val += img[y2,j]

            blr[i,j] = val/weight

    return blr