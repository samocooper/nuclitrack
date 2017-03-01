import numpy as np
import h5py

from kivy.uix.widget import Widget
from kivy.uix.slider import Slider
from kivy.uix.button import Button
from kivy.uix.gridlayout import GridLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.core.window import Window
from functools import partial

from skimage import filters
from skimage import morphology
from skimage.feature import peak_local_max
from scipy import ndimage

from .Image_widget import ImDisplay
from numpy_to_image import scale_im

# Standardize images using scale_im function

def im_standard(m):

    dims = m.shape
    m = m.flatten()

    im = scale_im(m, 1)
    im = np.reshape(im, dims)

    return im

# Batch Segmentation

def segment_im(param,im):

    im1 = clipping(im, param[0])  # perform image analysis operation
    im2 = background(im1, param[1])
    im3 = blur(im2, param[2])
    im_bin_uf = threshold(im3, param[3])
    im_b = object_filter(im_bin_uf, param[4])
    cell_center = cell_centers(im3, im_b, param[5])
    markers = fg_markers(cell_center, im_b, param[6])
    im_edge = sobel_edges(im1, param[7])
    labels = watershed(markers, im_b, im_edge, param[8])

    return labels

# Functions for segmentation


def clipping(im, val):

    im_temp = im.copy()

    if val != 0:
        im_temp[im > val] = val

    return im_temp


def background(im, val):

    im_temp = im.copy()

    if val != 0:
        im_temp = im_temp - filters.gaussian(im_temp, int(val))

    return im_temp


def blur(im, val):

    im_temp = im.copy()

    if val != 0:
        im_temp = filters.gaussian(im_temp, int(val))


    im_temp = im_standard(im_temp)

    return im_temp

def threshold(im, val):

    im_bin = im > val

    return im_bin


def object_filter(im_bin, val):

    im_bin = morphology.remove_small_objects(im_bin, val)

    return im_bin


def cell_centers(im, im_bin, val):

    d_mat = ndimage.distance_transform_edt(im_bin)
    d_mat = im_standard(d_mat)

    im_cent = (1 - val) * im + val * d_mat
    im_cent[np.invert(im_bin)] = 0

    return im_cent


def fg_markers(im_cent,im_bin,val):

    local_maxi = peak_local_max(im_cent, indices=False, min_distance=int(val), labels=im_bin, exclude_border=False)
    local_maxi[0, 0] = True
    markers = ndimage.label(local_maxi)[0]

    k = morphology.octagon(2, 2)
    markers = morphology.dilation(markers, selem=k)

    return markers


def sobel_edges(im,val):

    if val != 0:

        im = filters.gaussian(im, int(val))

    im = filters.sobel(im) + 1
    im = im_standard(im)

    return im


def watershed(markers,im_bin,im_edge,val):

    k = morphology.octagon(10, 10)
    bgm = morphology.binary_dilation(im_bin, selem=k)
    bgm[0, 0] = True

    d_mat = ndimage.distance_transform_edt(im_bin)
    d_mat = im_standard(d_mat)

    markers_temp = markers + np.invert(bgm)
    shed_im = (1 - val) * im_edge - val * d_mat

    labels = morphology.watershed(shed_im, markers_temp) - 1

    return labels


class SegmentationUI(Widget):

    def frame_select(self, instance, val):

        self.im = self.channel_im[int(val)].copy()
        self.im_disp.update_im(self.im)

    def segment_script(self, instance, val, **kwargs):

        # Dynamically update segmentation image and parameters

        state = kwargs['state']

        if state != 0:

            if state >= 2:

                if state == 2:  # if state is equal to stage of segmentation modify parameter

                    self.params[0] = val

                self.im1 = clipping(self.im, self.params[0])  # perform image analysis operation

                if state == 2:  # if state is equal to stage of segmentation update display image
                    self.im_disp.update_im(self.im1)

            if state >= 3:

                if state == 3:
                    self.params[1] = val

                self.im2 = background(self.im1, self.params[1])

                if state == 3:
                    self.im_disp.update_im(self.im2)

            if state >= 4:

                if state == 4:
                    self.params[2] = val

                self.im3 = blur(self.im2, self.params[2])

                if state == 4:
                    self.im_disp.update_im(self.im3)

            if state >= 5:

                if state == 5:
                    self.params[3] = val

                self.im_bin_uf = threshold(self.im3, self.params[3])

                if state == 5:
                    self.im_disp.update_im(self.im_bin_uf.astype(float))

            if state >= 6:

                if state == 6:
                    self.params[4] = val

                self.im_bin = object_filter(self.im_bin_uf, self.params[4])

                if state == 6:
                    self.im_disp.update_im(self.im_bin.astype(float))

            if state >= 7:

                if state == 7:
                    self.params[5] = val

                self.cell_center = cell_centers(self.im3, self.im_bin, self.params[5])

                if state == 7:
                    self.im_disp.update_im(self.cell_center)

            if state >= 8:

                if state == 8:
                    self.params[6] = val

                self.markers = fg_markers(self.cell_center, self.im_bin, self.params[6])

                if state == 8:
                    self.im_disp.update_im(self.cell_center + (self.markers > 0))

            if state == 1 or 9:

                if state == 1:
                    self.params[7] = val

                self.im_edge = sobel_edges(self.im1, self.params[7])

                if state == 1:
                    self.im_disp.update_im(self.im_edge)

            if state == 9:

                self.params[8] = val
                self.labels = watershed(self.markers, self.im_bin, self.im_edge, self.params[8])
                self.im_disp.update_im(self.labels.astype(float))

    def save_params(self, instance):

        self.parent.s_param['seg_param'][:] = self.params[:]

    def chan_select1(self,instance):
        self.channel_im = self.channel_1

    def chan_select2(self, instance):
        self.channel_im = self.channel_2

    def chan_select3(self, instance):
        self.channel_im = self.channel_3

    def initialize(self, channel_1, channel_2, channel_3, frames):

        self.parent.progression_state(3)

        self.state = 0

        self.channel_im = channel_1
        self.channel_1 = channel_1

        if channel_2:
            self.channel_2 = channel_2
        if channel_3:
            self.channel_3 = channel_3

        self.frames = frames

        self.parent.s_param.require_dataset('seg_param', (9,), dtype='f')
        self.params = self.parent.s_param['seg_param'][:]

        self.s_layout = FloatLayout(size=(Window.width, Window.height))

        self.im_disp = ImDisplay(size_hint=(.75, .7), pos_hint={'x': .2, 'y': .2})
        self.s_layout.add_widget(self.im_disp)

        self.im = self.channel_im[0]
        self.im_disp.create_im(self.im, 'PastelHeat')

        # Frame slider

        self.frame_slider = Slider(min=0, max=self.frames - 1, value=1, size_hint=(.3, .1), pos_hint={'x': .2, 'y': .9})
        self.frame_slider.bind(value=self.frame_select)

        if channel_2:

            self.ch1 = Button(text='Ch 1', size_hint=(.1, .05), pos_hint={'x': .55, 'y': .9})
            self.ch1.bind(on_press=self.chan_select1)

            self.ch2 = Button(text='Ch 2', size_hint=(.1, .05), pos_hint={'x': .7, 'y': .9})
            self.ch2.bind(on_press=self.chan_select2)

        if channel_3:

            self.ch3 = Button(text='Ch 1', size_hint=(.1, .05), pos_hint={'x': .55, 'y': .9})
            self.ch3.bind(on_press=self.chan_select3)

        # Sliders for updating parameters

        layout2 = GridLayout(cols=1, padding=2, size_hint=(.12, .8), pos_hint={'x': .01, 'y': .18})

        s0 = Slider(min=float(self.parent.ch1_min), max=float(self.parent.ch1_max),
                    step=float((self.parent.ch1_max-self.parent.ch1_min)/100), value=float(self.params[0]))
        s1 = Slider(min=0, max=300, step=5, value=float(self.params[1]))
        s2 = Slider(min=0, max=10, step=1, value=float(self.params[2]))
        s3 = Slider(min=0, max=1, step=0.01, value=float(self.params[3]))
        s4 = Slider(min=0, max=200, step=10, value=float(self.params[4]))
        s5 = Slider(min=0, max=1, step=0.05, value=float(self.params[5]))
        s6 = Slider(min=5, max=50, step=2, value=float(self.params[6]))
        s7 = Slider(min=0, max=10, step=1, value=float(self.params[7]))
        s8 = Slider(min=0, max=1, step=0.05, value=float(self.params[8]))
        b2 = Button(text='save params')

        s0.bind(value=partial(self.segment_script, state=2))
        s1.bind(value=partial(self.segment_script, state=3))
        s2.bind(value=partial(self.segment_script, state=4))
        s3.bind(value=partial(self.segment_script, state=5))
        s4.bind(value=partial(self.segment_script, state=6))
        s5.bind(value=partial(self.segment_script, state=7))
        s6.bind(value=partial(self.segment_script, state=8))
        s7.bind(value=partial(self.segment_script, state=1))
        s8.bind(value=partial(self.segment_script, state=9))
        b2.bind(on_press=self.save_params)

        layout2.add_widget(s0)
        layout2.add_widget(s1)
        layout2.add_widget(s2)
        layout2.add_widget(s3)
        layout2.add_widget(s4)
        layout2.add_widget(s5)
        layout2.add_widget(s6)
        layout2.add_widget(s7)
        layout2.add_widget(s8)
        layout2.add_widget(b2)

        self.layout2 = layout2

        with self.canvas:

            self.add_widget(self.s_layout)

            if channel_2:
                self.s_layout.add_widget(self.ch1)
                self.s_layout.add_widget(self.ch2)

            if channel_3:
                self.s_layout.add_widget(self.ch3)

            self.s_layout.add_widget(self.frame_slider)
            self.s_layout.add_widget(self.layout2)

    def remove(self):

        # Remove segmentation ui widgets

        self.layout2.clear_widgets()
        self.s_layout.clear_widgets()
        self.remove_widget(self.s_layout)

    def update_size(self, window, width, height):

        self.s_layout.width = width
        self.s_layout.height = height
