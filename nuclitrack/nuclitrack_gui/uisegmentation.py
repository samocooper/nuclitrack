import multiprocessing
import os
from functools import partial
from multiprocessing import Pool

import numpy as np
from kivy.core.window import Window
from kivy.graphics import Ellipse, Color
from kivy.uix.button import Button
from kivy.uix.dropdown import DropDown
from kivy.uix.filechooser import FileChooserListView
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.uix.progressbar import ProgressBar
from kivy.uix.slider import Slider
from kivy.uix.textinput import TextInput
from kivy.uix.togglebutton import ToggleButton
from kivy.uix.widget import Widget
from skimage.external import tifffile

from nuclitrack.kivy_wrappers.imagewidget import ImDisplay
from nuclitrack.nuclitrack_tools import segmentimages
from nuclitrack.kivy_wrappers import guitools

# Batch Segmentation


class BatchSegment(Widget):

    def __init__(self, movie, labels, params, parallel, seg_training=None, **kwargs):
        super().__init__(**kwargs)

        self.movie = movie
        self.labels = labels

        self.params = params
        self.layout = FloatLayout(size=(Window.width, Window.height))

        if self.params[11] == 1:
            self.clf = segmentimages.train_clf(seg_training)

        if parallel:

            self.seg_message = Label(text='[b][color=000000]Parallel Processing' \
                                          '\n   No Loading Bar[/b][/color]', markup=True,
                                     size_hint=(.2, .05), pos_hint={'x': .4, 'y': .65})
            with self.canvas:
                self.add_widget(self.layout)
                self.layout.add_widget(self.seg_message)

        else:

            self.seg_message = Label(text='[b][color=000000]Segmenting Images[/b][/color]', markup=True,
                                     size_hint=(.2, .05), pos_hint={'x': .4, 'y': .65})

            self.layout2 = GridLayout(rows=1, padding=2, size_hint=(.9, .1), pos_hint={'x': .05, 'y': .5})
            self.pb = ProgressBar(max=1000, size_hint=(8., 1.), pos_hint={'x': .1, 'y': .6}, value=1000/self.movie.frames)
            self.layout2.add_widget(self.pb)

            with self.canvas:
                self.add_widget(self.layout)
                self.layout.add_widget(self.seg_message)
                self.layout.add_widget(self.layout2)

    def update_bar(self, dt):
        self.pb.value += 1000/self.movie.frames

    def segment_im(self, frame, dt):

        if self.params[11] == 1:
            self.labels[frame, :, :] = segmentimages.segment_image(self.movie, self.params, self.clf, frame)
        else:
            self.labels[frame, :, :] = segmentimages.segment_image(self.movie, self.params, [], frame)

    def segment_parallel(self):

        cpu_count = multiprocessing.cpu_count()
        pool = Pool(cpu_count)
        if self.params[11] == 1:
            labels = pool.map(partial(segmentimages.segment_image, self.movie, self.params, self.clf),
                              range(self.movie.frames))
        else:
            labels = pool.map(partial(segmentimages.segment_image, self.movie, self.params, []),
                              range(self.movie.frames))

        pool.close()
        pool.join()

        for i in range(self.movie.frames):
            self.labels[i, :, :] = labels[i]

    def get(self):
        self.seg_message.text = '[b][color=000000]Images Segmented[/b][/color]'
        return self.labels

    def update_size(self, window, width, height):

        self.width = width
        self.height = height

# Segmentation UI

class LabelWindow(Widget):

    pixel_list_fg = []
    pixel_list_bg = []
    level = -1

    def paint(self, touch):

        x_size = self.size[0]
        y_size = self.size[1]
        xpos = touch.pos[0]
        ypos = touch.pos[1]

        xpos_norm = (touch.pos[0] - self.pos[0]) / x_size
        ypos_norm = 1 - ((touch.pos[1] - self.pos[1]) / y_size)

        ds = self.parent.parent.brush_size / 1000

        with self.canvas:
            if 0 < xpos_norm < 1 and 0 < ypos_norm < 1:
                if self.level == 0:

                    Color(0, 1, 0, 0.3)
                    self.pixel_list_fg.append([xpos_norm, ypos_norm, ds/2, ds/2])
                    self.dot = Ellipse(size=(ds*x_size, ds*y_size), pos=(xpos - ds*x_size / 2, ypos - ds*y_size / 2))
                    Color(1, 1, 1, 1)

                if self.level == 1:

                    Color(1, 0, 0, 0.3)
                    self.pixel_list_bg.append([xpos_norm, ypos_norm, ds/2, ds/2])
                    self.dot = Ellipse(size=(ds*x_size, ds*y_size), pos=(xpos - ds*x_size / 2, ypos - ds*y_size / 2))
                    Color(1, 1, 1, 1)

    def on_touch_down(self, touch):
        self.paint(touch)

    def on_touch_move(self, touch):
        self.paint(touch)

    def clear(self, instance):
        self.canvas.clear()
        self.pixel_list_bg = []
        self.pixel_list_fg = []

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

def unique_pixls(X):

    X_mask = X[:, 0] * 10 ** 5 + X[:, 1]
    X_mask = np.unique(X_mask)
    X_filter = np.zeros((X_mask.shape[0], 2))
    X_filter[:, 0] = X_mask // 10 ** 5
    X_filter[:, 1] = X_mask % 10 ** 5
    X_filter = X_filter.astype(int)

    return X_filter

class SegmentationUI(Widget):

    def __init__(self, movie, params, training=None, **kwargs):
        super().__init__(**kwargs)

        self.current_state = 0
        self.current_frame = 0
        self.movie = movie
        self.params = params

        self.seg_channels = self.params[15:].astype(int)
        print(self.seg_channels)

        self.state = 0

        self.s_layout = FloatLayout(size=(Window.width, Window.height))

        self.im_disp = ImDisplay(size_hint=(.76, .76), pos_hint={'x': .23, 'y': .14})
        self.s_layout.add_widget(self.im_disp)

        self.mov_disp = ImDisplay(size_hint=(.2, .2), pos_hint={'x': .78, 'y': .14})
        self.s_layout.add_widget(self.mov_disp)

        self.im = movie.read_im(0, 0)
        self.im_disp.create_im(self.im, 'PastelHeat')
        self.mov_disp.create_im(self.im, 'PastelHeat')

        # Frame slider

        self.frame_slider = guitools.frame_slider(self.movie.frames, self.change_frame,
                                                  size_hint=(.29, .06), pos_hint={'x': .23, 'y': .91})
        self.s_layout.add_widget(self.frame_slider)

        self.parallel_button = ToggleButton(text=' Multiple Cores ',
                                size_hint=(.15, .04), pos_hint={'x': .682, 'y': .923}, markup=True)
        self.parallel_button.bind(on_press=self.update_parallel)
        self.s_layout.add_widget(self.parallel_button)

        self.edge_button = ToggleButton(text=' Filter edge ',
                                            size_hint=(.15, .04), pos_hint={'x': .53, 'y': .923}, markup=True)
        self.edge_button.bind(on_press=self.update_edge)
        self.s_layout.add_widget(self.edge_button)

        if self.params[9] == 1:
            self.edge_button.state = 'down'

        # Drop down menu for choosing which channel

        self.channel_choice = DropDown()

        for i in range(self.movie.channels):

            channel_btn = ToggleButton(text='Channel ' + str(i + 1), size_hint_y=None)
            channel_btn.bind(on_press=partial(self.change_channel, i))
            self.channel_choice.add_widget(channel_btn)

        self.main_button = Button(text=' Channel ', size_hint=(.15, .04), pos_hint={'x': .834, 'y': .923}, markup=True)
        self.main_button.bind(on_release=self.channel_choice.open)
        self.channel_choice.bind(on_select=lambda instance, x: setattr(self.main_button, 'text', x))
        self.s_layout.add_widget(self.main_button)

        # Sliders for updating parameters

        layout2 = GridLayout(cols=1, padding=2, size_hint=(.2, .84), pos_hint={'x': .01, 'y': .14})

        self.b1 = ToggleButton(text='ML_segment')

        s1 = Slider(min=0, max=1, step=0.002, value=float(self.params[0]))
        s2 = Slider(min=0, max=300, step=5, value=float(self.params[1]))
        s3 = Slider(min=0, max=10, step=1, value=float(self.params[2]))
        s4 = Slider(min=0, max=1, step=0.005, value=float(self.params[3]))
        s5 = Slider(min=0, max=200, step=10, value=float(self.params[4]))
        s6 = Slider(min=0, max=1, step=0.05, value=float(self.params[5]))
        s7 = Slider(min=0, max=50, step=2, value=float(self.params[6]))
        s8 = Slider(min=0, max=10, step=1, value=float(self.params[7]))
        s9 = Slider(min=0, max=1, step=0.05, value=float(self.params[8]))
        b2 = Button(text='Save Params')

        self.b1.bind(on_press=self.ml_segment)

        s1.bind(value=partial(self.segment_script, state=2))
        s2.bind(value=partial(self.segment_script, state=3))
        s3.bind(value=partial(self.segment_script, state=4))
        s4.bind(value=partial(self.segment_script, state=5))
        s5.bind(value=partial(self.segment_script, state=6))
        s6.bind(value=partial(self.segment_script, state=7))
        s7.bind(value=partial(self.segment_script, state=8))
        s8.bind(value=partial(self.segment_script, state=1))
        s9.bind(value=partial(self.segment_script, state=9))

        b2.bind(on_press=self.save_params)

        self.s1_label = Label(text='[color=000000]Clipping Limit: ' + str(self.params[0]) + '[/color]', markup=True)
        self.s2_label = Label(text='[color=000000]Background Blur: ' + str(self.params[1]) + '[/color]', markup=True)
        self.s3_label = Label(text='[color=000000]Image Blur: ' + str(self.params[2]) + '[/color]', markup=True)
        self.s4_label = Label(text='[color=000000]Threshold: ' + str(self.params[3]) + '[/color]', markup=True)
        self.s5_label = Label(text='[color=000000]Smallest Object: ' + str(self.params[4]) + '[/color]', markup=True)
        self.s6_label = Label(text='[color=000000]Distance to Intensity: ' + str(self.params[5]) + '[/color]',
                              markup=True)
        self.s7_label = Label(text='[color=000000]Separation Distance: ' + str(self.params[6]) + '[/color]',
                              markup=True)
        self.s8_label = Label(text='[color=000000]Edge Blur: ' + str(self.params[7]) + '[/color]', markup=True)
        self.s9_label = Label(text='[color=000000]Watershed Ratio: ' + str(self.params[8]) + '[/color]', markup=True)

        self.spacer = Label(text='')

        layout2.add_widget(s1)
        layout2.add_widget(self.s1_label)
        layout2.add_widget(s2)
        layout2.add_widget(self.s2_label)
        layout2.add_widget(s3)
        layout2.add_widget(self.s3_label)
        layout2.add_widget(self.b1)
        layout2.add_widget(self.spacer)
        layout2.add_widget(s4)
        layout2.add_widget(self.s4_label)
        layout2.add_widget(s5)
        layout2.add_widget(self.s5_label)
        layout2.add_widget(s6)
        layout2.add_widget(self.s6_label)
        layout2.add_widget(s7)
        layout2.add_widget(self.s7_label)
        layout2.add_widget(s8)
        layout2.add_widget(self.s8_label)
        layout2.add_widget(s9)
        layout2.add_widget(self.s9_label)
        layout2.add_widget(b2)

        self.layout2 = layout2

        if self.params[0] > 0:
            self.segment_script([], self.params[0], state=2)
        if self.params[1] > 0:
            self.segment_script([], self.params[1], state=3)
        if self.params[2] > 0:
            self.segment_script([], self.params[2], state=4)

        self.imml = self.im.copy()

        if self.params[11] == 1:
            if self.params[11] == 1:
                self.clf = segmentimages.train_clf(training)
                self.class_flag = True

        self.ml_mode = False

        with self.canvas:

            self.add_widget(self.s_layout)
            self.s_layout.add_widget(self.layout2)

    def ml_segment(self, instance):

        self.level = -1
        self.label_window = LabelWindow(size_hint=(.76, .76), pos_hint={'x': .23, 'y': .14})
        self.s_layout.add_widget(self.label_window)

        self.s_layout.remove_widget(self.layout2)
        self.layout3 = GridLayout(cols=1, padding=2, size_hint=(.2, .84), pos_hint={'x': .01, 'y': .14})

        self.brush_size = 30
        btn_size = Slider(min=1, max=100, step=1, value=30)
        btn_size.bind(value=self.change_size)
        self.layout3.add_widget(btn_size)
        self.brush_lbl = Label(text='[color=000000]Brush Size ' + str(self.brush_size) + '[/color]', markup=True)
        self.layout3.add_widget(self.brush_lbl)

        select_fg = ToggleButton(text='Select Nuclei', group='label_data')
        select_fg.bind(on_press=partial(self.label_im, level=0))
        self.layout3.add_widget(select_fg)

        select_bg = ToggleButton(text='Select Background', group='label_data')
        select_bg.bind(on_press=partial(self.label_im, level=1))
        self.layout3.add_widget(select_bg)

        clear = Button(text='Clear Selection')
        clear.bind(on_press=self.label_window.clear)
        self.layout3.add_widget(clear)

        classify = Button(text='Calculate Probability')
        classify.bind(on_press=self.classify)
        self.layout3.add_widget(classify)

        self.params[12] = 0
        soc = Slider(min=0, max=20, step=1, value=0)
        soc.bind(value=self.open_close)
        self.layout3.add_widget(soc)
        self.oc_lbl = Label(text='[color=000000]Open Close ' + str(self.params[12]) + '[/color]', markup=True)
        self.layout3.add_widget(self.oc_lbl)

        reset = Button(text='Reset Training')
        reset.bind(on_press=self.reset_train)
        self.layout3.add_widget(reset)

        cont = Button(text='Continue Segmentation')
        cont.bind(on_press=self.continue_seg)
        self.layout3.add_widget(cont)
        self.ml_mode = True
        self.s_layout.add_widget(self.layout3)


        self.train_count = Label(text='[color=000000]Train pxls ' + str(0) + '[/color]', markup=True)

        for g in self.parent.params:
            if g == 'seg_training':
                count = self.parent.params['seg_training']['X'].shape[0]
                self.train_count.text = '[color=000000]Train pxls ' + str(count) + '[/color]'

        self.layout3.add_widget(self.train_count)

    def open_close(self, instance, val):

        if self.params[11] == 1 and val > 0:

            self.oc_lbl.text = '[color=000000]Open Close ' + str(np.round(val)) + '[/color]'
            self.im_open_close = segmentimages.open_close(self.im_class, val)
            self.im_disp.update_im(self.im_open_close)
            self.params[12] = val

        else:
            self.params[12] = val

    def reset_train(self, instance):
        for g in self.parent.params:
            if g == 'seg_training':
                del self.parent.params['seg_training']
                self.params[11] = 0
                self.class_flag = False
        self.train_count.text = '[color=000000]Train pxls ' + str(0) + '[/color]'
        self.im_disp.update_im(self.im)

    def continue_seg(self, instance):

        self.ml_mode = False
        self.s_layout.remove_widget(self.layout3)
        self.s_layout.remove_widget(self.label_window)
        self.s_layout.add_widget(self.layout2)

    def change_size(self, instance, val):

        self.brush_size = val
        self.brush_lbl.text = '[color=000000]Brush Size ' + str(np.round(val)) + '[/color]'

    def classify(self, instance):

        print(self.imml.shape)

        self.params[13] = 12
        self.params[14] = 2

        wsize = int(self.params[13])
        stride = int(self.params[14])

        P = np.zeros((1, 2))
        N = np.zeros((1, 2))

        pixel_bg = self.label_window.pixel_list_bg
        pixel_fg = self.label_window.pixel_list_fg

        if len(pixel_fg) > 0 and len(pixel_bg) > 0:

            pixel_bg = np.asarray(pixel_bg)
            pixel_fg = np.asarray(pixel_fg)

            pixel_bg[:, [0, 2]] = pixel_bg[:, [0, 2]] * self.imml.shape[1]
            pixel_bg[:, [1, 3]] = pixel_bg[:, [1, 3]] * self.imml.shape[0]

            pixel_fg[:, [0, 2]] = pixel_fg[:, [0, 2]] * self.imml.shape[1]
            pixel_fg[:, [1, 3]] = pixel_fg[:, [1, 3]] * self.imml.shape[0]

            pixel_bg = pixel_bg.astype(int)
            pixel_fg = pixel_fg.astype(int)

            for i in range(pixel_bg.shape[0]):

                pixls = ellipse_roi(pixel_bg[i, :], self.imml.shape)
                N = np.vstack((N, pixls))

            for i in range(pixel_fg.shape[0]):
                pixls = ellipse_roi(pixel_fg[i, :], self.imml.shape)
                P = np.vstack((P, pixls))

            vinds = np.arange(self.imml.shape[0], self.imml.shape[0]-self.params[13]+1, -1)-1
            hinds = np.arange(self.imml.shape[1], self.imml.shape[1]-self.params[13]+1, -1)-1
            rinds = np.arange(self.params[13]+1, 0, -1)

            vinds = vinds.astype(int)
            hinds = hinds.astype(int)
            rinds = rinds.astype(int)

            conv_im_temp = np.vstack((self.imml[rinds, :], self.imml, self.imml[vinds, :]))
            conv_im = np.hstack((conv_im_temp[:, rinds],  conv_im_temp,  conv_im_temp[:, hinds]))

            P = P[1:, :] + self.params[13]+1
            N = N[1:, :] + self.params[13]+1

            P = unique_pixls(P)
            N = unique_pixls(N)

            P = P.astype(int)
            N = N.astype(int)

            X = []
            y = []

            for i in range(P.shape[0]):
                roi = conv_im[P[i, 0]-wsize:P[i, 0]+wsize, P[i, 1]-wsize:P[i, 1]+wsize]
                roi = roi[::stride, ::stride]
                X.append(roi.flatten())
                y.append(1)

            for i in range(N.shape[0]):
                roi = conv_im[N[i, 0]-wsize:N[i, 0]+wsize, N[i, 1]-wsize:N[i, 1]+wsize]
                roi = roi[::stride, ::stride]
                X.append(roi.flatten())
                y.append(0)

            X = np.asarray(X)
            y = np.asarray(y)

            R = np.random.permutation(np.arange(X.shape[0]))
            R = R.astype(int)

            X = X[R, :]
            y = y[R]

            self.label_window.clear([])

        # Create if no training data else append

        training_flag = False

        for g in self.parent.params:
            if g == 'seg_training':
               training_flag = True

        if training_flag or (len(pixel_fg) > 0 and len(pixel_bg) > 0):
            if training_flag:

                X1 = self.parent.params['seg_training']['X'][...]
                y1 = self.parent.params['seg_training']['y'][...]

                if len(pixel_fg) > 0 and len(pixel_bg) > 0:

                    X = np.vstack((X, X1))
                    y = np.hstack((y, y1))

                else:

                    X = X1
                    y = y1

                del self.parent.params['seg_training']

                self.train_count.text = '[color=000000]Train pxls ' + str(X.shape[0]) + '[/color]'
                self.training_hdf5 = self.parent.params.create_group('seg_training')
                self.training_hdf5.create_dataset('X', data=X)
                self.training_hdf5.create_dataset('y', data=y)

            else:

                self.train_count.text = '[color=000000]Train pxls ' + str(X.shape[0]) + '[/color]'
                self.training_hdf5 = self.parent.params.create_group('seg_training')
                self.training_hdf5.create_dataset('X', data=X)
                self.training_hdf5.create_dataset('y', data=y)

            self.clf = segmentimages.train_clf(self.training_hdf5)
            self.class_flag = True
            self.im_class = segmentimages.im_probs(self.imml, self.clf, wsize, stride)

            self.im_open_close = self.im_class.copy()
            self.params[11] = 1
            self.im_disp.update_im(self.im_class)

        else:

            self.parent.error_message('Error in classifier: no training data selected')

    def label_im(self, instance, level):

        self.label_window.level = level


    def segment_script(self, instance, val, **kwargs):

        # Dynamically update segmentation image and parameters

        state = kwargs['state']
        self.current_state = state

        if state != 0:

            if state >= 2:

                if state == 2:  # if state is equal to stage of segmentation modify parameter
                    if not val == -1:
                        self.params[0] = val
                        self.s1_label.text = '[color=000000]Clipping Limit ' + str(np.round(val, 2)) + '[/color]'

                self.im1 = segmentimages.clipping(self.im, self.params[0])  # perform image analysis operation
                self.imml = self.im1.copy()

                if state == 2:  # if state is equal to stage of segmentation update display image
                    self.im_disp.update_im(self.im1)

            if state >= 3:

                if state == 3:
                    if not val == -1:
                        self.params[1] = val
                        self.s2_label.text = '[color=000000]Background blur: ' + str(np.round(val, 2)) + '[/color]'

                self.im2 = segmentimages.background(self.im1, self.params[1])
                self.imml = self.im2.copy()

                if state == 3:
                    self.im_disp.update_im(self.im2)

            if state >= 4:

                if state == 4:
                    if not val == -1:
                        self.params[2] = val
                        self.s3_label.text = '[color=000000]Image blur: ' + str(np.round(val, 2)) + '[/color]'

                self.im3 = segmentimages.blur(self.im2, self.params[2])
                self.imml = self.im3.copy()

                if state == 4:
                    self.im_disp.update_im(self.im3)

            if state >= 5:

                if state == 5:
                    if not val == -1:
                        self.params[3] = val
                        self.s4_label.text = '[color=000000]Threshold: ' + str(np.round(val, 2)) + '[/color]'

                if self.params[11] == 1:
                    if self.class_flag:
                        self.im_class = segmentimages.im_probs(self.im3, self.clf, int(self.params[13]), int(self.params[14]))
                        self.im_open_close = segmentimages.open_close(self.im_class, self.params[12])
                        self.class_flag = False
                    self.im_bin_uf = segmentimages.threshold(self.im_open_close, self.params[3])

                else:
                    self.im_bin_uf = segmentimages.threshold(self.im3, self.params[3])

                if state == 5:
                    self.im_disp.update_im(self.im_bin_uf.astype(float))

            if state >= 6:

                if state == 6:
                    if not val == -1:
                        self.params[4] = val
                        self.s5_label.text = '[color=000000]Smallest Object: ' + str(np.round(val, 2)) + '[/color]'

                self.im_bin = segmentimages.object_filter(self.im_bin_uf, self.params[4])

                if state == 6:
                    self.im_disp.update_im(self.im_bin.astype(float))

            if state >= 7:

                if state == 7:
                    if not val == -1:
                        self.params[5] = val
                        self.s6_label.text = '[color=000000]Distance to Intensity: ' + str(np.round(val, 2)) + '[/color]'

                if self.params[11] == 1:
                    [self.cell_center, self.d_mat] = segmentimages.cell_centers(self.im_open_close, self.im_bin, self.params[5])

                else:
                    [self.cell_center, self.d_mat] = segmentimages.cell_centers(self.im3, self.im_bin, self.params[5])

                if state == 7:
                    self.im_disp.update_im(self.cell_center)

            if state >= 8:

                if state == 8:
                    if not val == -1:
                        self.params[6] = val
                        self.s7_label.text = '[color=000000]Separation Distance: ' + str(np.round(val, 2)) + '[/color]'

                self.markers = segmentimages.fg_markers(self.cell_center, self.im_bin, self.params[6], self.params[9])

                if state == 8:
                    self.im_disp.update_im(self.cell_center + (self.markers > 0))

            if state == 1 or 9:

                if state == 1:
                    if not val == -1:
                        self.params[7] = val
                        self.s8_label.text = '[color=000000]Edge Blur: ' + str(np.round(val, 2)) + '[/color]'

                self.im_edge = segmentimages.sobel_edges(self.im1, self.params[7])

                if state == 1:
                    self.im_disp.update_im(self.im_edge)

            if state == 9:
                if not val == -1:
                    self.params[8] = val
                    self.s9_label.text = '[color=000000]Watershed Ratio: ' + str(np.round(val, 2)) + '[/color]'

                self.labels = segmentimages.watershed(self.markers, self.im_bin, self.im_edge, self.d_mat, self.params[8], self.params[9])


                self.im_disp.update_im(self.labels.astype(float))

    def save_params(self, instance):

        self.parent.params['seg_param'][...] = self.params[...]

    def update_parallel(self, instance):
        if instance.state == 'down':
            self.parent.parallel = True
        else:
            self.parent.parallel = False

    def update_edge(self, instance):

        if instance.state == 'down':
            self.params[9] = 1
        else:
            self.params[9] = 0

    def change_frame(self, val):

        self.current_frame = val
        self.update_im()

    def change_channel(self, val, instance):

        if instance.state == 'down':
            self.seg_channels[val] = 1
            self.params[15+val] = 1
        else:
            if sum(self.seg_channels) > 1:
                self.seg_channels[val] = 0
                self.params[15 + val] = 0

        self.update_im()

    def update_im(self):

        self.im = self.movie.comb_im(self.seg_channels, self.current_frame)
        self.imml = self.im.copy()

        if self.current_state > 0:
            self.segment_script([], -1, state=self.current_state)
        else:
            self.im_disp.update_im(self.im)

        self.mov_disp.update_im(self.im)

        if self.ml_mode:
            self.label_window.clear([])

        self.class_flag = True

    def update_size(self, window, width, height):

        self.s_layout.width = width
        self.s_layout.height = height

class ViewSegment(Widget):

    def __init__(self, movie, labels, **kwargs):
        super().__init__(**kwargs)

        self.movie = movie
        self.labels = labels

        self.view = 'seg_view'
        self.layout_choice = GridLayout(cols=3, size_hint=(.45, .05), pos_hint={'x': .52, 'y': .925})
        self.view_button = ToggleButton(text='View Labels', group='view')
        self.export_button = ToggleButton(text='Export Labels', group='view')
        self.ring_button = ToggleButton(text='Ring Region')

        self.view_button.bind(on_press=self.view_segment)
        self.export_button.bind(on_press=self.export_data)
        self.ring_button.bind(on_press=self.ring_toggle)

        self.layout_choice.add_widget(self.view_button)
        self.layout_choice.add_widget(self.export_button)
        self.layout_choice.add_widget(self.ring_button)

        self.s_layout = FloatLayout(size=(Window.width, Window.height))
        self.s_layout.add_widget(self.layout_choice)

        im_temp = self.labels[0, :, :]

        #### SEG WIDGETS ####

        self.im_disp = ImDisplay(size_hint=(.8, .73), pos_hint={'x': .1, 'y': .14})
        self.im_disp.create_im(im_temp, 'Random')

        self.frame_slider = Slider(min=0, max=self.movie.frames - 1, value=1, size_hint=(.3, .06),
                                   pos_hint={'x': .1, 'y': .94})
        self.frame_slider.bind(value=self.segment_frame)
        self.frame_text = Label(text='[color=000000]Frame: ' + str(0) + '[/color]',
                                size_hint=(.19, .04), pos_hint={'x': .155, 'y': .9}, markup=True)

        self.frame_minus = Button(text='<<',
                                  size_hint=(.05, .03), pos_hint={'x': .1, 'y': .905}, markup=True)
        self.frame_plus = Button(text='>>',
                                 size_hint=(.05, .03), pos_hint={'x': .35, 'y': .905}, markup=True)
        self.frame_minus.bind(on_press=self.frame_backward)
        self.frame_plus.bind(on_press=self.frame_forward)

        #### FILE WIDGETS ####

        self.file_choose = FileChooserListView(size_hint=(.9, .6), pos_hint={'x': .05, 'y': .14})

        # Crate dir
        self.folder_input = TextInput(text='Directory Name',
                                      multiline=False, size_hint=(.65, .05), pos_hint={'x': .31, 'y': .85})
        self.folder_input.bind(on_text_validate=self.make_folder)
        self.folder_label = Label(text='[b][color=000000]Create directory name: [/b][/color]', markup=True,
                                  size_hint=(.25, .05), pos_hint={'x': .05, 'y': .85})

        # Export labels to tif files

        self.text_file_input = TextInput(text='File Name',
                                         multiline=False, size_hint=(.65, .05), pos_hint={'x': .31, 'y': .79})
        self.text_file_input.bind(on_text_validate=self.export_files)
        self.file_label = Label(text='[b][color=000000]Choose file name: [/b][/color]', markup=True,
                                size_hint=(.25, .05), pos_hint={'x': .05, 'y': .79})

        self.s_layout.add_widget(self.frame_text)
        self.s_layout.add_widget(self.im_disp)
        self.s_layout.add_widget(self.frame_slider)
        self.s_layout.add_widget(self.frame_plus)
        self.s_layout.add_widget(self.frame_minus)

        with self.canvas:
            self.add_widget(self.s_layout)

    def ring_toggle(self, instance):

        if instance.state == 'down':
            self.parent.ring_flag = True
        else:
            self.parent.ring_flag = False

    def make_folder(self, instance):

        os.makedirs(os.path.join(self.file_choose.path, instance.text))
        self.folder_input.text = 'Directory made, re-enter present dir to view it'

    def export_files(self, instance):

        temp_path = self.file_choose.path
        digits = len(str(self.movie.frames))

        for i in range(self.movie.frames):
            num = str(i)
            num = num.zfill(digits)
            fname = os.path.join(temp_path, instance.text + '_' + num + '.tif')
            tifffile.imsave(fname, self.labels[i, :, :])

        self.text_file_input.text = 'Images written, re-enter present dir to view them'

    def export_data(self, instance):

        if self.view == 'seg_view':

            self.s_layout.remove_widget(self.im_disp)
            self.s_layout.remove_widget(self.frame_slider)
            self.s_layout.remove_widget(self.frame_plus)
            self.s_layout.remove_widget(self.frame_minus)
            self.s_layout.remove_widget(self.frame_text)

            self.s_layout.add_widget(self.file_choose)
            self.s_layout.add_widget(self.text_file_input)
            self.s_layout.add_widget(self.file_label)
            self.s_layout.add_widget(self.folder_input)
            self.s_layout.add_widget(self.folder_label)

        self.view = 'file_view'

    def view_segment(self, instance):

        if self.view == 'file_view':

            self.s_layout.remove_widget(self.file_choose)
            self.s_layout.remove_widget(self.text_file_input)
            self.s_layout.remove_widget(self.file_label)
            self.s_layout.remove_widget(self.folder_input)
            self.s_layout.remove_widget(self.folder_label)

            self.s_layout.add_widget(self.im_disp)
            self.s_layout.add_widget(self.frame_slider)
            self.s_layout.add_widget(self.frame_text)
            self.s_layout.add_widget(self.frame_plus)
            self.s_layout.add_widget(self.frame_minus)

        self.view = 'seg_view'

    def frame_forward(self, instance):
        if self.frame_slider.value < self.movie.frames - 1:
            self.frame_slider.value += 1

    def frame_backward(self, instance):
        if self.frame_slider.value > 0:
            self.frame_slider.value -= 1

    def segment_frame(self, instance, val):

        self.frame_text.text = '[color=000000]Frame: ' + str(int(val)) + '[/color]'
        im_temp = self.labels[int(val), :, :]
        self.im_disp.update_im(im_temp)

    def update_size(self, window, width, height):

        self.s_layout.width = width
        self.s_layout.height = height

