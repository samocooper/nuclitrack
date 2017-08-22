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

from ..nuclitrack_guitools.imagewidget import ImDisplay
from ..nuclitrack_tools import segmentimages
from ..nuclitrack_guitools import guitools
from ..nuclitrack_tools import classifypixels


class BatchSegment(Widget):

    def __init__(self, movie, labels, params, parallel, **kwargs):
        super().__init__(**kwargs)

        self.params = params['seg_param'][...]

        # Classifier is used if training data is present
        self.clf = 0
        if 'seg_training' in params:
            self.clf = classifypixels.train_clf(params['seg_training'])

        self.movie = movie
        self.labels = labels
        self.layout = FloatLayout(size=(Window.width, Window.height))

        if parallel:

            # Cannot provide loading bar if parallel processing is used with current structure

            self.seg_message = Label(text='[b][color=000000]Parallel Processing' \
                                          '\n   No Loading Bar[/b][/color]', markup=True,
                                     size_hint=(.2, .05), pos_hint={'x': .4, 'y': .65})
            with self.canvas:

                self.add_widget(self.layout)
                self.layout.add_widget(self.seg_message)

        else:

            # Add loading bar to canvas and initiate scheduling of segmentation

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

        self.labels[frame, :, :] = segmentimages.segment_image(self.movie, self.params, self.clf, frame)

    def segment_parallel(self):

        # Schedule segmentation of images in parallel using pool class

        cpu_count = multiprocessing.cpu_count()
        pool = Pool(cpu_count)

        labels = pool.map(partial(segmentimages.segment_image, self.movie, self.params, self.clf),
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


class LabelWindow(Widget):

    # lists for storing locations and sizes of brush strokes for background and foreground

    pixel_list_fg = []
    pixel_list_bg = []

    def paint(self, touch):

        # Get location of clicks

        x_size = self.size[0]
        y_size = self.size[1]

        xpos = touch.pos[0]
        ypos = touch.pos[1]

        # Scale to between 0 and 1 relative to image size

        xpos_norm = (touch.pos[0] - self.pos[0]) / x_size
        ypos_norm = 1 - ((touch.pos[1] - self.pos[1]) / y_size)

        ds = self.parent.parent.brush_size / 1000

        with self.canvas:

            # Mark location of clicks with ellipses (circles scaled to image), gives impression of painting

            if 0 < xpos_norm < 1 and 0 < ypos_norm < 1:
                if self.parent.parent.select_fg.state == 'down':

                    Color(0, 1, 0, 0.3)
                    self.pixel_list_fg.append([xpos_norm, ypos_norm, ds/2, ds/2])
                    self.dot = Ellipse(size=(ds*x_size, ds*y_size), pos=(xpos - ds*x_size / 2, ypos - ds*y_size / 2))
                    Color(1, 1, 1, 1)

                if self.parent.parent.select_bg.state == 'down':

                    Color(1, 0, 0, 0.3)
                    self.pixel_list_bg.append([xpos_norm, ypos_norm, ds/2, ds/2])
                    self.dot = Ellipse(size=(ds*x_size, ds*y_size), pos=(xpos - ds*x_size / 2, ypos - ds*y_size / 2))
                    Color(1, 1, 1, 1)

    def on_touch_down(self, touch):
        self.paint(touch)

    def on_touch_move(self, touch):
        self.paint(touch)

    def clear(self, instance):

        # Clear canvas and pixel lists

        self.canvas.clear()
        self.pixel_list_bg = []
        self.pixel_list_fg = []


class SegmentationUI(Widget):

    def __init__(self, movie, params, **kwargs):
        super().__init__(**kwargs)

        # Movie object

        self.movie = movie

        # How far the user has progressed through segmentation

        self.current_state = 0
        self.prior_state = 0

        # Current frame

        self.frame = 0

        # Parameters for segmentation and channels to segment on

        self.params = params['seg_param'][...]
        self.seg_channels = self.params[15:18].astype(int)

        # Set classifier, label window and classified image to None to prevent them being prematurely used

        self.clf = None
        self.label_window = None
        self.im_class = None

        # Master layout for segmentation UI

        self.s_layout = FloatLayout(size=(Window.width, Window.height))

        # Add image widgets to visualize segmentation results and the original image

        self.im_disp = ImDisplay(size_hint=(.76, .76), pos_hint={'x': .23, 'y': .14})
        self.s_layout.add_widget(self.im_disp)
        self.mov_disp = ImDisplay(size_hint=(.2, .2), pos_hint={'x': .78, 'y': .14})
        self.s_layout.add_widget(self.mov_disp)

        self.im = movie.read_im(0, 0)
        self.im_disp.create_im(self.im, 'PastelHeat')
        self.mov_disp.create_im(self.im, 'PastelHeat')

        # Add frame slider widget for choosing frames

        self.frame_slider = guitools.FrameSlider(self.movie.frames, self.change_frame,
                                                 size_hint=(.29, .06), pos_hint={'x': .23, 'y': .91})
        self.s_layout.add_widget(self.frame_slider)

        # Add button for optional segmentation using multiple cores

        self.parallel_button = ToggleButton(text=' Multiple Cores ',
                                            size_hint=(.15, .04), pos_hint={'x': .682, 'y': .923})
        self.parallel_button.bind(on_press=self.update_parallel)
        self.s_layout.add_widget(self.parallel_button)

        #  Add button to optionally filter out segments touching the edge

        self.edge_button = ToggleButton(text=' Filter edge ', size_hint=(.15, .04), pos_hint={'x': .53, 'y': .923})
        self.edge_button.bind(on_press=self.update_edge)
        self.s_layout.add_widget(self.edge_button)

        if self.params[9] == 1:
            self.edge_button.state = 'down'

        # Drop down menu for choosing which channel to segment on

        self.channel_choice = DropDown()

        for i in range(self.movie.channels):
            channel_btn = ToggleButton(text='Channel ' + str(i + 1), size_hint_y=None)

            if self.params[15+i] == 1:
                channel_btn.state = 'down'

            channel_btn.bind(on_press=partial(self.change_channel, i))
            self.channel_choice.add_widget(channel_btn)

        self.main_button = Button(text=' Channel ', size_hint=(.15, .04), pos_hint={'x': .834, 'y': .923})
        self.main_button.bind(on_release=self.channel_choice.open)
        self.channel_choice.bind(on_select=lambda instance, x: setattr(self.main_button, 'text', x))
        self.s_layout.add_widget(self.main_button)

        # Sliders for updating parameters

        layout2 = GridLayout(cols=1, padding=2, size_hint=(.2, .84), pos_hint={'x': .01, 'y': .14})

        s1 = Slider(min=0, max=1, step=0.002, value=float(self.params[0]))
        s2 = Slider(min=0, max=300, step=5, value=float(self.params[1]))
        s3 = Slider(min=0, max=10, step=1, value=float(self.params[2]))
        s4 = Slider(min=0, max=1, step=0.005, value=float(self.params[3]))
        s5 = Slider(min=0, max=200, step=10, value=float(self.params[4]))
        s6 = Slider(min=0, max=1, step=0.05, value=float(self.params[5]))
        s7 = Slider(min=0, max=50, step=2, value=float(self.params[6]))
        s8 = Slider(min=0, max=10, step=1, value=float(self.params[7]))
        s9 = Slider(min=0, max=1, step=0.02, value=float(self.params[8]))

        s1.bind(value=partial(self.segment_script, state=2))
        s2.bind(value=partial(self.segment_script, state=3))
        s3.bind(value=partial(self.segment_script, state=4))
        s4.bind(value=partial(self.segment_script, state=5))
        s5.bind(value=partial(self.segment_script, state=6))
        s6.bind(value=partial(self.segment_script, state=7))
        s7.bind(value=partial(self.segment_script, state=8))
        s8.bind(value=partial(self.segment_script, state=1))
        s9.bind(value=partial(self.segment_script, state=9))

        # Slider labels

        self.s1_label = guitools.ntlabel(text='Clipping Limit: ' + str(self.params[0]), style=2)
        self.s2_label = guitools.ntlabel(text='Background Blur: ' + str(self.params[1]), style=2)
        self.s3_label = guitools.ntlabel(text='Image Blur: ' + str(self.params[2]), style=2)
        self.s4_label = guitools.ntlabel(text='Threshold: ' + str(self.params[3]), style=2)
        self.s5_label = guitools.ntlabel(text='Smallest Object: ' + str(self.params[4]), style=2)
        self.s6_label = guitools.ntlabel(text='Distance to Intensity: ' + str(self.params[5]), style=2)
        self.s7_label = guitools.ntlabel(text='Separation Distance: ' + str(self.params[6]), style=2)
        self.s8_label = guitools.ntlabel(text='Edge Blur: ' + str(self.params[7]), style=2)
        self.s9_label = guitools.ntlabel(text='Watershed Ratio: ' + str(self.params[8]), style=2)

        # Button for using pixel classifier on images

        self.b1 = ToggleButton(text='Classify Pixels')
        self.b1.bind(on_press=self.ml_segment)
        self.spacer = Label(text='')

        # Button to save parameters for segmentation

        b2 = Button(text='Save Params')
        b2.bind(on_press=self.save_params)

        # Add widgets to layout

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

        with self.canvas:

            self.add_widget(self.s_layout)
            self.s_layout.add_widget(self.layout2)

    def ml_segment(self, instance):

        # Ensure that image is at correct stage of segmentation

        self.segment_script([], self.params[2], state=4)

        # Swap layouts to ml layout

        self.s_layout.remove_widget(self.layout2)
        self.ml_layout = GridLayout(cols=1, padding=2, size_hint=(.2, .84), pos_hint={'x': .01, 'y': .14})

        # Add widget that allows user to paint foreground and background pixels, stores list of central pixel and brush
        # size for each brush stroke

        self.label_window = LabelWindow(size_hint=(.76, .76), pos_hint={'x': .23, 'y': .14})
        self.s_layout.add_widget(self.label_window)

        # Set default size of paint brush and attach widget to adjust this size

        self.brush_size = 20

        btn_size = Slider(min=1, max=100, step=1, value=30)
        btn_size.bind(value=self.change_size)
        self.ml_layout.add_widget(btn_size)

        self.brush_lbl = Label(text='[color=000000]Brush Size ' + str(self.brush_size) + '[/color]', markup=True)
        self.ml_layout.add_widget(self.brush_lbl)

        # Toggle buttons to select whether foreground or background is being labelled

        self.select_fg = ToggleButton(text='Select Nuclei', group='label_data')
        self.ml_layout.add_widget(self.select_fg)

        self.select_bg = ToggleButton(text='Select Background', group='label_data')
        self.ml_layout.add_widget(self.select_bg)

        # Button to clear current selection

        clear = Button(text='Clear Selection')
        clear.bind(on_press=self.label_window.clear)
        self.ml_layout.add_widget(clear)

        # Button to add training examples to list of all training data and output probability map

        classify = Button(text='Calculate Probability')
        classify.bind(on_press=self.classify)
        self.ml_layout.add_widget(classify)

        # Button to restore image and if the user wants to add more training data

        rev_image = Button(text='Revert Image')
        rev_image.bind(on_press=self.revert_image)
        self.ml_layout.add_widget(rev_image)

        # Open close parameter to reduce noise in pixel level classification

        soc = Slider(min=0, max=20, step=1, value=int(self.params[12]))
        soc.bind(value=self.open_close)
        self.ml_layout.add_widget(soc)

        self.oc_lbl = Label(text='[color=000000]Open Close ' + str(self.params[12]) + '[/color]', markup=True)
        self.ml_layout.add_widget(self.oc_lbl)

        # Reset all training data

        reset = Button(text='Reset Training')
        reset.bind(on_press=self.reset_train)
        self.ml_layout.add_widget(reset)

        # Leave the ml window and return to normal segmentation

        cont = Button(text='Continue Segmentation')
        cont.bind(on_press=self.continue_seg)
        self.ml_layout.add_widget(cont)

        # Counter which gives the number of training examples for pixel level classification

        self.train_count = Label(text='[color=000000]Train pxls ' + str(0) + '[/color]', markup=True)

        if 'seg_training' in self.parent.params:
            count = self.parent.params['seg_training']['X'].shape[0]
            self.train_count.text = '[color=000000]Train pxls ' + str(count) + '[/color]'

        self.ml_layout.add_widget(self.train_count)

        # Add the ml_layout

        self.s_layout.add_widget(self.ml_layout)

    def reset_train(self, instance):

        if 'seg_training' in self.parent.params:
            del self.parent.params['seg_training']

        self.train_count.text = '[color=000000]Train pxls ' + str(0) + '[/color]'
        self.clf = None
        self.im_disp.update_im(self.im)

    def continue_seg(self, instance):

        self.s_layout.remove_widget(self.ml_layout)
        self.s_layout.remove_widget(self.label_window)
        self.s_layout.add_widget(self.layout2)
        self.label_window = None

    def change_size(self, instance, val):

        self.brush_size = val
        self.brush_lbl.text = '[color=000000]Brush Size ' + str(np.round(val)) + '[/color]'

    def classify(self, instance):

        self.segment_script([], self.params[2], state=4)
        dims = self.movie.dims

        # Parameters for stride and width of MLP region of interest

        self.params[13] = 12
        self.params[14] = 2

        wsize = int(self.params[13])
        stride = int(self.params[14])

        # Initialize arrays for foreground and background pixels

        P = np.zeros((1, 2))
        N = np.zeros((1, 2))

        pixel_bg = self.label_window.pixel_list_bg
        pixel_fg = self.label_window.pixel_list_fg

        if len(pixel_fg) > 0 and len(pixel_bg) > 0:

            # Scale pixel lists in range 0 to 1 to image dimensions

            pixel_bg = np.asarray(pixel_bg)
            pixel_fg = np.asarray(pixel_fg)

            pixel_bg[:, [0, 2]] = pixel_bg[:, [0, 2]] * dims[1]
            pixel_bg[:, [1, 3]] = pixel_bg[:, [1, 3]] * dims[0]

            pixel_fg[:, [0, 2]] = pixel_fg[:, [0, 2]] * dims[1]
            pixel_fg[:, [1, 3]] = pixel_fg[:, [1, 3]] * dims[0]

            pixel_bg = pixel_bg.astype(int)
            pixel_fg = pixel_fg.astype(int)

            # Identify pixels within elliptical roi around central pixel defined by brush size

            for i in range(pixel_bg.shape[0]):

                pixls = classifypixels.ellipse_roi(pixel_bg[i, :], dims)
                N = np.vstack((N, pixls))

            for i in range(pixel_fg.shape[0]):
                pixls = classifypixels.ellipse_roi(pixel_fg[i, :], dims)
                P = np.vstack((P, pixls))

            # Calculate necessary expansion of image needed to perform pixel convolution

            conv_im = segmentimages.expand_im(self.im3, self.params[13])

            # Add expansion amount to pixels in list to center on new image

            P = P[1:, :] + self.params[13]+1
            N = N[1:, :] + self.params[13]+1

            # Take unique pixel values

            P = classifypixels.unique_pixls(P)
            N = classifypixels.unique_pixls(N)

            P = P.astype(int)
            N = N.astype(int)

            X = []
            y = []

            # Extract intensity values for given pixels

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

            # Convert to numpy arrays for classification

            X = np.asarray(X)
            y = np.asarray(y)

            R = np.random.permutation(np.arange(X.shape[0]))
            R = R.astype(int)

            X = X[R, :]
            y = y[R]

            self.label_window.clear([])

        # Create new training data hdf5 file if no prior data exists else append

        if len(pixel_fg) > 0 and len(pixel_bg) > 0:
            if 'seg_training' in self.parent.params:

                X1 = self.parent.params['seg_training']['X'][...]
                y1 = self.parent.params['seg_training']['y'][...]

                if len(pixel_fg) > 0 and len(pixel_bg) > 0:

                    X = np.vstack((X, X1))
                    y = np.hstack((y, y1))

                else:

                    X = X1
                    y = y1

                del self.parent.params['seg_training']

                # Update training pixel counts

                self.train_count.text = '[color=000000]Train pxls ' + str(X.shape[0]) + '[/color]'
                training_hdf5 = self.parent.params.create_group('seg_training')
                training_hdf5.create_dataset('X', data=X)
                training_hdf5.create_dataset('y', data=y)

            else:

                self.train_count.text = '[color=000000]Train pxls ' + str(X.shape[0]) + '[/color]'
                training_hdf5 = self.parent.params.create_group('seg_training')
                training_hdf5.create_dataset('X', data=X)
                training_hdf5.create_dataset('y', data=y)

        # Perform classification is training data is present

        if 'seg_training' in self.parent.params:

            self.clf = classifypixels.train_clf(self.parent.params['seg_training'])
            self.im_class = segmentimages.im_probs(self.im3.copy(), self.clf, wsize, stride)

            # Also perform open and closing here if the parameter is greater than 0

            if self.params[12] > 0:
                self.im_open_close = segmentimages.open_close(self.im_class, self.params[12])
            else:
                self.im_open_close = self.im_class.copy()

            self.im_disp.update_im(self.im_open_close)

    def revert_image(self, instance):

        self.im_disp.update_im(self.im3)

    def open_close(self, instance, val):

        if self.im_class is None:
            self.im_class = self.im3  # users can call open close prior to classification as well

        if val > 0:

            self.oc_lbl.text = '[color=000000]Open Close ' + str(np.round(val)) + '[/color]'
            self.im_open_close = segmentimages.open_close(self.im_class, val)
            self.im_disp.update_im(self.im_open_close)
            self.params[12] = val

        else:
            self.params[12] = val

    def segment_script(self, instance, val, **kwargs):

        # Dynamically update segmentation image and parameters

        state = kwargs['state']
        self.current_state = state

        if state != 0:

            if state >= 2 >= self.prior_state or state == 2:

                if state == 2:  # if state is equal to stage of segmentation modify parameter
                    if not val == -1:
                        self.params[0] = val
                        guitools.ntchange(label=self.s1_label, text='Clipping Limit ' + str(np.round(val, 2)), style=2)

                self.im1 = segmentimages.clipping(self.im, self.params[0])  # perform image analysis operation

                if state == 2:  # if state is equal to stage of segmentation update display image

                    self.im_disp.update_im(self.im1)
                    self.prior_state = 2

            if state >= 3 >= self.prior_state or state == 3:

                if state == 3:
                    if not val == -1:
                        self.params[1] = val
                        guitools.ntchange(label=self.s2_label, text='Background blur: ' + str(np.round(val, 2)), style=2)

                self.im2 = segmentimages.background(self.im1, self.params[1])

                if state == 3:

                    self.im_disp.update_im(self.im2)
                    self.prior_state = 3

            if state >= 4 >= self.prior_state or state == 4:

                if state == 4:
                    if not val == -1:
                        self.params[2] = val
                        guitools.ntchange(label=self.s3_label, text='Image blur: ' + str(np.round(val, 2)), style=2)

                self.im3 = segmentimages.blur(self.im2, self.params[2])

                if state == 4:
                    self.im_disp.update_im(self.im3)
                    self.prior_state = 4

            # Pixel classifier state sits optionally between blurring and threshold

            if state >= 4.5 >= self.prior_state:

                if self.clf is None:
                    if 'seg_training' in self.parent.params:
                        self.clf = classifypixels.train_clf(self.parent.params['seg_training'])

                if self.clf is not None:
                    self.im3b = segmentimages.im_probs(self.im3, self.clf, int(self.params[13]), int(self.params[14]))
                else:
                    self.im3b = self.im3

                if self.params[12] > 0:

                    self.im3c = segmentimages.open_close(self.im3b, self.params[12])
                else:
                    self.im3c = self.im3b

                self.prior_state = 4.5

            if state >= 5 >= self.prior_state or state == 5:

                if state == 5:
                    if not val == -1:
                        self.params[3] = val
                        guitools.ntchange(label=self.s4_label, text='Threshold: ' + str(np.round(val, 2)), style=2)

                self.im_bin_uf = segmentimages.threshold(self.im3c, self.params[3])

                if state == 5:
                    self.im_disp.update_im(self.im_bin_uf.astype(float))
                    self.prior_state = 5

            if state >= 6 >= self.prior_state or state == 6:

                if state == 6:
                    if not val == -1:
                        self.params[4] = val
                        guitools.ntchange(label=self.s5_label, text='Smallest Object: ' + str(np.round(val, 2)), style=2)

                self.im_bin = segmentimages.object_filter(self.im_bin_uf, self.params[4])

                if state == 6:
                    self.im_disp.update_im(self.im_bin.astype(float))
                    self.prior_state = 6

            if state >= 7 >= self.prior_state or state == 7:

                if state == 7:
                    if not val == -1:
                        self.params[5] = val
                        guitools.ntchange(label=self.s6_label, text='Distance to Intensity: ' + str(np.round(val, 2)), style=2)

                if self.params[11] == 1:
                    [self.cell_center, self.d_mat] = segmentimages.cell_centers(self.im_open_close, self.im_bin, self.params[5])

                else:
                    [self.cell_center, self.d_mat] = segmentimages.cell_centers(self.im3, self.im_bin, self.params[5])

                if state == 7:
                    self.im_disp.update_im(self.cell_center)
                    self.prior_state = 7

            if state >= 8 >= self.prior_state or state == 8:

                if state == 8:
                    if not val == -1:
                        self.params[6] = val
                        guitools.ntchange(label=self.s7_label, text='Separation Distance: ' + str(np.round(val, 2)), style=2)

                self.markers = segmentimages.fg_markers(self.cell_center, self.im_bin, self.params[6], self.params[9])

                if state == 8:
                    self.im_disp.update_im(self.cell_center + (self.markers > 0))
                    self.prior_state = 8

            if state == 1 or 9:

                if state == 1:
                    if not val == -1:
                        self.params[7] = val
                        guitools.ntchange(label=self.s8_label, text='Edge Blur: ' + str(np.round(val, 2)), style=2)

                self.im_edge = segmentimages.sobel_edges(self.im1, self.params[7])

                if state == 1:
                    self.im_disp.update_im(self.im_edge)

            if state == 9:
                if not val == -1:
                    self.params[8] = val
                    guitools.ntchange(label=self.s9_label, text='Watershed Ratio: ' + str(np.round(val, 2)), style=2)

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

        self.frame = val
        self.update_im()
        self.prior_state = 0

    def change_channel(self, val, instance):

        if instance.state == 'down':
            self.seg_channels[val] = 1
            self.params[15+val] = 1
        else:
            if sum(self.seg_channels) > 1:
                self.seg_channels[val] = 0
                self.params[15 + val] = 0

        self.update_im()
        self.prior_state = 0

    def update_im(self):

        self.im = self.movie.comb_im(self.seg_channels, self.frame)
        self.imml = self.im.copy()

        if self.current_state > 0:
            self.segment_script([], -1, state=self.current_state)
        else:
            self.im_disp.update_im(self.im)

        self.mov_disp.update_im(self.im)

        if self.label_window is not None:
            self.label_window.clear([])

    def update_size(self, window, width, height):

        self.s_layout.width = width
        self.s_layout.height = height


class ViewSegment(Widget):

    def __init__(self, movie, labels, **kwargs):
        super().__init__(**kwargs)

        self.movie = movie
        self.labels = labels

        self.view = 'seg_view'

        # Buttons for choosing between whether to view labelled segments or export labelled images

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

        self.label_layout = FloatLayout(size=(Window.width, Window.height))
        self.save_layout = FloatLayout(size=(Window.width, Window.height))

        im_temp = self.labels[0, :, :]

        # Label Viewing Widgets

        self.im_disp = ImDisplay(size_hint=(.8, .73), pos_hint={'x': .1, 'y': .14})
        self.im_disp.create_im(im_temp, 'Random')

        self.frame_slider = guitools.FrameSlider(self.movie.frames, self.change_frame,
                                                 size_hint=(.37, .06), pos_hint={'x': .13, 'y': .91})

        self.label_layout.add_widget(self.frame_slider)
        self.label_layout.add_widget(self.im_disp)
        self.label_layout.add_widget(self.layout_choice)

        # File Viewing Widgets

        self.file_choose = FileChooserListView(size_hint=(.9, .6), pos_hint={'x': .05, 'y': .14})
        self.save_layout.add_widget(self.file_choose)

        # Widgets for creating directory

        self.folder_input = TextInput(text='Directory Name',
                                      multiline=False, size_hint=(.65, .05), pos_hint={'x': .31, 'y': .85})
        self.folder_input.bind(on_text_validate=self.make_folder)
        self.folder_label = Label(text='[b][color=000000]Create directory name: [/b][/color]', markup=True,
                                  size_hint=(.25, .05), pos_hint={'x': .05, 'y': .85})

        self.save_layout.add_widget(self.folder_input)
        self.save_layout.add_widget(self.folder_label)

        # Widgets for exporting labels to tif files

        self.text_file_input = TextInput(text='File Name',
                                         multiline=False, size_hint=(.65, .05), pos_hint={'x': .31, 'y': .79})
        self.text_file_input.bind(on_text_validate=self.export_files)
        self.file_label = Label(text='[b][color=000000]Choose file name: [/b][/color]', markup=True,
                                size_hint=(.25, .05), pos_hint={'x': .05, 'y': .79})

        self.save_layout.add_widget(self.text_file_input)
        self.save_layout.add_widget(self.file_label)

        with self.canvas:
            self.add_widget(self.label_layout)

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

            self.label_layout.remove_widget(self.layout_choice)
            self.save_layout.add_widget(self.layout_choice)

            self.remove_widget(self.label_layout)
            self.add_widget(self.save_layout)

        self.view = 'file_view'

    def view_segment(self, instance):

        if self.view == 'file_view':

            self.save_layout.remove_widget(self.layout_choice)
            self.label_layout.add_widget(self.layout_choice)

            self.remove_widget(self.save_layout)
            self.add_widget(self.label_layout)

        self.view = 'seg_view'

    def change_frame(self, val):

        im_temp = self.labels[int(val), :, :]
        self.im_disp.update_im(im_temp)

    def update_size(self, window, width, height):

        self.label_layout.width = width
        self.label_layout.height = height
        self.save_layout.width = width
        self.save_layout.height = height

    def ring_toggle(self, instance):

        # Toggle button for whether to extract intensity of ring region in feature extraction

        if instance.state == 'down':
            self.parent.ring_flag = True
        else:
            self.parent.ring_flag = False

