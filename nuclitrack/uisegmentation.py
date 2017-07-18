import numpy as np
import multiprocessing
from multiprocessing import Pool
from PIL import Image
from skimage.external import tifffile
import os

from kivy.uix.widget import Widget
from kivy.uix.slider import Slider
from kivy.uix.button import Button
from kivy.uix.gridlayout import GridLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.core.window import Window
from kivy.uix.togglebutton import ToggleButton
from kivy.uix.filechooser import FileChooserListView
from kivy.uix.textinput import TextInput
from kivy.uix.label import Label
from kivy.uix.dropdown import DropDown
from kivy.uix.progressbar import ProgressBar
from functools import partial

from . import segmentimages
from .imagewidget import ImDisplay


# Batch Segmentation

class BatchSegment(Widget):

    def __init__(self, file_list, min_val, max_val, labels, params, frames, parallel, **kwargs):
        super().__init__(**kwargs)

        self.frames = frames
        self.file_list = file_list
        self.labels = labels
        self.min_val = min_val
        self.max_val = max_val

        self.params = params
        self.layout = FloatLayout(size=(Window.width, Window.height))

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
            self.pb = ProgressBar(max=1000, size_hint=(8., 1.), pos_hint={'x': .1, 'y': .6}, value=1000/self.frames)
            self.layout2.add_widget(self.pb)

            with self.canvas:
                self.add_widget(self.layout)
                self.layout.add_widget(self.seg_message)
                self.layout.add_widget(self.layout2)

    def update_bar(self, dt):
        self.pb.value += 1000/self.frames

    def segment_im(self, frame, dt):

        self.labels[frame, :, :] = segmentimages.segment_image(self.params, self.min_val, self.max_val,
                                                               self.file_list[int(frame)])

    def segment_parallel(self):

        cpu_count = multiprocessing.cpu_count()
        pool = Pool(cpu_count)
        labels = pool.map(partial(segmentimages.segment_image, self.params, self.min_val, self.max_val), self.file_list)
        pool.close()
        pool.join()

        for i in range(self.frames):
            self.labels[i, :, :] = labels[i]

    def get(self):
        self.seg_message.text = '[b][color=000000]Images Segmented[/b][/color]'
        return self.labels

# Segmentation UI

class SegmentationUI(Widget):
    def __init__(self, file_list, min_vals, max_vals, frames, channels, params, **kwargs):
        super().__init__(**kwargs)

        self.params = params
        self.current_frame = 0
        self.channels = channels
        self.frames = frames

        self.file_list = file_list
        self.min_vals = min_vals
        self.max_vals = max_vals

        self.seg_channel = int(self.params[10])
        self.state = 0

        self.s_layout = FloatLayout(size=(Window.width, Window.height))

        self.im_disp = ImDisplay(size_hint=(.76, .76), pos_hint={'x': .23, 'y': .14})
        self.s_layout.add_widget(self.im_disp)

        self.mov_disp = ImDisplay(size_hint=(.2, .2), pos_hint={'x': .78, 'y': .14})
        self.s_layout.add_widget(self.mov_disp)

        self.im = np.asarray(Image.open(file_list[self.seg_channel][0]))
        self.im = self.im.astype(float)
        self.im -= self.min_vals[self.seg_channel]
        self.im /= self.max_vals[self.seg_channel]

        self.im_disp.create_im(self.im, 'PastelHeat')
        self.mov_disp.create_im(self.im, 'PastelHeat')

        # Frame slider

        self.frame_slider = Slider(min=0, max=self.frames - 1, value=1, size_hint=(.3, .06), pos_hint={'x': .23, 'y': .94})
        self.frame_slider.bind(value=self.frame_select)
        self.frame_text = Label(text='[color=000000]Frame: ' + str(0) + '[/color]',
                                size_hint=(.2, .04), pos_hint={'x': .28, 'y': .9}, markup=True)

        self.frame_minus = Button(text='<<',
                                size_hint =(.05, .03), pos_hint={'x': .23, 'y': .905}, markup=True)
        self.frame_plus = Button(text='>>',
                                 size_hint=(.05, .03), pos_hint={'x': .48, 'y': .905}, markup=True)
        self.frame_minus.bind(on_press=self.frame_backward)
        self.frame_plus.bind(on_press=self.frame_forward)

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

        for i in range(self.channels):
            channel_btn = ToggleButton(text='Channel ' + str(i + 1), group='channel', size_hint_y=None)
            channel_btn.bind(on_press=partial(self.change_channel, i))
            self.channel_choice.add_widget(channel_btn)

        self.main_button = Button(text=' Channel ', size_hint=(.15, .04), pos_hint={'x': .834, 'y': .923}, markup=True)
        self.main_button.bind(on_release=self.channel_choice.open)
        self.channel_choice.bind(on_select=lambda instance, x: setattr(self.main_button, 'text', x))
        self.s_layout.add_widget(self.main_button)

        # Sliders for updating parameters

        layout2 = GridLayout(cols=1, padding=2, size_hint=(.2, .84), pos_hint={'x': .01, 'y': .14})

        s0 = Slider(min=0, max=1, step=0.002, value=float(self.params[0]))
        s1 = Slider(min=0, max=300, step=5, value=float(self.params[1]))
        s2 = Slider(min=0, max=10, step=1, value=float(self.params[2]))
        s3 = Slider(min=0, max=1, step=0.005, value=float(self.params[3]))
        s4 = Slider(min=0, max=200, step=10, value=float(self.params[4]))
        s5 = Slider(min=0, max=1, step=0.05, value=float(self.params[5]))
        s6 = Slider(min=0, max=50, step=2, value=float(self.params[6]))
        s7 = Slider(min=0, max=10, step=1, value=float(self.params[7]))
        s8 = Slider(min=0, max=1, step=0.05, value=float(self.params[8]))
        b2 = Button(text='Save Params')

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

        layout2.add_widget(s0)
        layout2.add_widget(self.s1_label)
        layout2.add_widget(s1)
        layout2.add_widget(self.s2_label)
        layout2.add_widget(s2)
        layout2.add_widget(self.s3_label)
        layout2.add_widget(s3)
        layout2.add_widget(self.s4_label)
        layout2.add_widget(s4)
        layout2.add_widget(self.s5_label)
        layout2.add_widget(s5)
        layout2.add_widget(self.s6_label)
        layout2.add_widget(s6)
        layout2.add_widget(self.s7_label)
        layout2.add_widget(s7)
        layout2.add_widget(self.s8_label)
        layout2.add_widget(s8)
        layout2.add_widget(self.s9_label)
        layout2.add_widget(b2)

        self.layout2 = layout2

        with self.canvas:

            self.add_widget(self.s_layout)

            self.s_layout.add_widget(self.frame_slider)
            self.s_layout.add_widget(self.frame_plus)
            self.s_layout.add_widget(self.frame_minus)

            self.s_layout.add_widget(self.frame_text)
            self.s_layout.add_widget(self.layout2)

    def frame_forward(self, instance):
        if self.frame_slider.value < self.frames - 1:
            self.frame_slider.value += 1

    def frame_backward(self, instance):
        if self.frame_slider.value > 0:
            self.frame_slider.value -= 1

    def frame_select(self, instance, val):

        self.current_frame = int(val)

        self.im = np.asarray(Image.open(self.file_list[self.seg_channel][int(val)]))
        self.im = self.im.astype(float)
        self.im -= self.min_vals[self.seg_channel]
        self.im /= self.max_vals[self.seg_channel]

        self.im_disp.update_im(self.im)
        self.mov_disp.update_im(self.im)
        self.frame_text.text = '[color=000000]Frame: ' + str(int(val)) + '[/color]'

    def segment_script(self, instance, val, **kwargs):

        # Dynamically update segmentation image and parameters

        state = kwargs['state']

        if state != 0:

            if state >= 2:

                if state == 2:  # if state is equal to stage of segmentation modify parameter

                    self.params[0] = val
                    self.s1_label.text = '[color=000000]Clipping Limit ' + str(np.round(val, 2)) + '[/color]'

                self.im1 = segmentimages.clipping(self.im, self.params[0])  # perform image analysis operation

                if state == 2:  # if state is equal to stage of segmentation update display image
                    self.im_disp.update_im(self.im1)

            if state >= 3:

                if state == 3:
                    self.params[1] = val
                    self.s2_label.text = '[color=000000]Background blur: ' + str(np.round(val, 2)) + '[/color]'

                self.im2 = segmentimages.background(self.im1, self.params[1])

                if state == 3:
                    self.im_disp.update_im(self.im2)

            if state >= 4:

                if state == 4:
                    self.params[2] = val
                    self.s3_label.text = '[color=000000]Image blur: ' + str(np.round(val, 2)) + '[/color]'

                self.im3 = segmentimages.blur(self.im2, self.params[2])

                if state == 4:
                    self.im_disp.update_im(self.im3)

            if state >= 5:

                if state == 5:
                    self.params[3] = val
                    self.s4_label.text = '[color=000000]Threshold: ' + str(np.round(val, 2)) + '[/color]'

                self.im_bin_uf = segmentimages.threshold(self.im3, self.params[3])

                if state == 5:
                    self.im_disp.update_im(self.im_bin_uf.astype(float))

            if state >= 6:

                if state == 6:
                    self.params[4] = val
                    self.s5_label.text = '[color=000000]Smallest Object: ' + str(np.round(val, 2)) + '[/color]'

                self.im_bin = segmentimages.object_filter(self.im_bin_uf, self.params[4])

                if state == 6:
                    self.im_disp.update_im(self.im_bin.astype(float))

            if state >= 7:

                if state == 7:
                    self.params[5] = val
                    self.s6_label.text = '[color=000000]Distance to Intensity: ' + str(np.round(val, 2)) + '[/color]'

                [self.cell_center, self.d_mat] = segmentimages.cell_centers(self.im3, self.im_bin, self.params[5])

                if state == 7:
                    self.im_disp.update_im(self.cell_center)

            if state >= 8:

                if state == 8:
                    self.params[6] = val
                    self.s7_label.text = '[color=000000]Separation Distance: ' + str(np.round(val, 2)) + '[/color]'

                self.markers = segmentimages.fg_markers(self.cell_center, self.im_bin, self.params[6],self.params[9])

                if state == 8:
                    self.im_disp.update_im(self.cell_center + (self.markers > 0))

            if state == 1 or 9:

                if state == 1:
                    self.params[7] = val
                    self.s8_label.text = '[color=000000]Edge Blur: ' + str(np.round(val, 2)) + '[/color]'

                self.im_edge = segmentimages.sobel_edges(self.im1, self.params[7])

                if state == 1:
                    self.im_disp.update_im(self.im_edge)

            if state == 9:

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

    def change_channel(self, val, instance):

        self.seg_channel = int(val)

        self.im = np.asarray(Image.open(self.file_list[self.seg_channel][self.current_frame]))
        self.im = self.im.astype(float)
        self.im -= self.min_vals[self.seg_channel]
        self.im /= self.max_vals[self.seg_channel]

        self.im_disp.update_im(self.im)
        self.mov_disp.update_im(self.im)
        self.params[10] = val

    def update_size(self, window, width, height):

        self.s_layout.width = width
        self.s_layout.height = height


class ViewSegment(Widget):

    def __init__(self, labels=None, frames=None, **kwargs):
        super().__init__(**kwargs)

        self.labels = labels
        self.frames = frames

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

        self.frame_slider = Slider(min=0, max=self.frames - 1, value=1, size_hint=(.3, .06),
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
        digits = len(str(self.frames))

        for i in range(self.frames):
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
        if self.frame_slider.value < self.frames - 1:
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

