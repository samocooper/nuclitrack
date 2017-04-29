import numpy as np
import platform
import h5py

import os
from functools import partial
from skimage.external import tifffile

from kivy.uix.dropdown import DropDown
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.uix.togglebutton import ToggleButton
from kivy.uix.label import Label
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.textinput import TextInput
from kivy.core.window import Window
from kivy.uix.filechooser import FileChooserListView
from kivy.uix.gridlayout import GridLayout

class LoadingUI(Widget):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Master layout for loading screen

        self.ld_layout = FloatLayout(size=(Window.width, Window.height))
        self.add_widget(self.ld_layout)

        self.img_layout = FloatLayout(size=(Window.width, Window.height))
        self.add_widget(self.img_layout)

        # Assign/Load HDF5 file for storing data from the current field of view.

        # Record whether both files are loaded
        self.file_loaded = [False, False]

        # Label
        self.label_fov = Label(text='[b][color=000000]Experimental Data File[/b][/color]', markup=True,
                               size_hint=(.48, .05), pos_hint={'x': .01, 'y': .95})
        self.ld_layout.add_widget(self.label_fov)

        # Input
        self.text_input_fov = TextInput(text='example_data.hdf5', multiline=False,
                                        size_hint=(.48, .05), pos_hint={'x': .01, 'y': .9})
        self.text_input_fov.bind(on_text_validate=partial(self.dir, 'fov'))
        self.ld_layout.add_widget(self.text_input_fov)

        # Display loaded file

        self.loaded_fov = Label(text='[b][color=000000] [/b][/color]', markup=True,
                                size_hint=(.49, .05), pos_hint={'x': .01, 'y': .85})
        self.ld_layout.add_widget(self.loaded_fov)

        # Assign/Load HDF5 file for storing tracking and segmentation parameter data.

        # Label
        self.label_param = Label(text='[b][color=000000]Parameter Data File[/b][/color]', markup=True,
                                 size_hint=(.48, .05), pos_hint={'x': .51, 'y': .95})
        self.ld_layout.add_widget(self.label_param)

        # Input
        self.text_input_param = TextInput(text='example_params.hdf5', multiline=False,
                                          size_hint=(.48, .05), pos_hint={'x': .51, 'y': .9})
        self.text_input_param.bind(on_text_validate=partial(self.dir, 'param'))
        self.ld_layout.add_widget(self.text_input_param)

        # Display loaded file
        self.loaded_param = Label(text='[b][color=000000] [/b][/color]', markup=True,
                                  size_hint=(.48, .05), pos_hint={'x': .51, 'y': .85})
        self.ld_layout.add_widget(self.loaded_param)

        # Info on file loading
        self.error_message = Label(text='[b][color=000000][/b][/color]', markup=True,
                                   size_hint=(.19, .05), pos_hint={'x': .75, 'y': .14})
        self.ld_layout.add_widget(self.error_message)

    def dir(self, input_type, obj):
        input_text = obj.text
        # Display loaded files

        if input_type == 'fov':

            self.parent.fov = h5py.File(input_text, "a")
            self.loaded_fov.text = '[b][color=000000] File loaded: ' + input_text + '[/b][/color]'
            self.file_loaded[0] = True

        if input_type == 'param':
            self.parent.params = h5py.File(input_text, "a")
            self.loaded_param.text = '[b][color=000000] File loaded: ' + input_text + '[/b][/color]'
            self.file_loaded[1] = True

        ########################
        # FILE LOADING OPTIONS #
        ########################

        if self.file_loaded[0] and self.file_loaded[1]:

            # Inform user if data already exists in file

            flag = True

            for g in self.parent.fov:
                if g == 'image_filenames':
                    self.message = Label(text='[b][color=000000] Data exists in the HDF5 file [/b][/color]',
                                              markup=True, size_hint=(.5, .05), pos_hint={'x': .25, 'y': .55})
                    self.ld_layout.add_widget(self.message)
                    flag = False

                    images = []

                    images_np = self.parent.fov['image_filenames'][...]
                    for i in range(images_np.shape[0]):
                        images.append([str(fname, encoding='utf8') for fname in images_np[i, :]])

                    self.load_movie(images)

            # Give user choice of how to load image series

            if flag:

                self.load_choice = GridLayout(rows=1, padding=2, size_hint=(.98, .05), pos_hint={'x': .01, 'y': .8})

                btn1 = ToggleButton(text='Load from file names', group='load_type')
                btn1.bind(on_press=partial(self.load_imgs, 'file'))
                self.load_choice.add_widget(btn1)

                btn2 = ToggleButton(text='Load from text file', group='load_type')
                btn2.bind(on_press=partial(self.load_imgs, 'text'))
                self.load_choice.add_widget(btn2)

                btn3 = ToggleButton(text='Load from directory', group='load_type')
                btn3.bind(on_press=partial(self.load_imgs, 'dir'))
                self.load_choice.add_widget(btn3)

                self.ld_layout.add_widget(self.load_choice)

            self.parent.progression[0] = 1
            self.parent.progression_state(2)

    def load_imgs(self, load_type, obj):

        self.img_layout.clear_widgets()

        #########################################
        # LOAD IMAGES FROM FIRST AND LAST NAMES #
        #########################################

        if load_type == 'file':

            self.max_channel = 3
            self.channel = 0
            self.img_pos = 0

            self.file_names = []
            for i in range((self.max_channel + 1)*2):
                self.file_names.append('')

            # Layout for file buttons

            self.series_choice = GridLayout(rows=1, padding=2, size_hint=(.98, .05), pos_hint={'x': .01, 'y': .755})

            # Drop down menu for choosing which channel
            self.channel_choice = DropDown()

            for i in range(self.max_channel):

                channel_btn = ToggleButton(text='Channel ' + str(i+1), group='channel', size_hint_y=None)
                channel_btn.bind(on_press=partial(self.change_channel, i))
                self.channel_choice.add_widget(channel_btn)

            channel_btn = ToggleButton(text='Labels', group='channel', size_hint_y=None)
            channel_btn.bind(on_press=partial(self.change_channel, self.max_channel))
            self.channel_choice.add_widget(channel_btn)

            self.main_button = Button(text='Select Channel')
            self.main_button.bind(on_release=self.channel_choice.open)
            self.channel_choice.bind(on_select=lambda instance, x: setattr(self.main_button, 'text', x))
            self.series_choice.add_widget(self.main_button)

            # Determine whether first or last file in file sequence will be selected

            first_image = ToggleButton(text='First Image', group='file_pos')
            first_image.bind(on_press=partial(self.image_pos, 0))
            self.series_choice.add_widget(first_image)

            last_image = ToggleButton(text='Last Image', group='file_pos')
            last_image.bind(on_press=partial(self.image_pos, 1))
            self.series_choice.add_widget(last_image)

            last_image = Button(text='Load Images')
            last_image.bind(on_press=self.auto_load)
            self.series_choice.add_widget(last_image)

            self.img_layout.add_widget(self.series_choice)

            # Labels for informing users which images have been loaded

            self.first_img_name = Label(text='[b][color=000000] [/b][/color]',
                                 markup=True, size_hint=(.44, .05), pos_hint={'x': .05, 'y': .7})
            self.last_img_name = Label(text='[b][color=000000] [/b][/color]',
                                 markup=True, size_hint=(.44, .05), pos_hint={'x': .51, 'y': .7})

            self.img_layout.add_widget(self.first_img_name)
            self.img_layout.add_widget(self.last_img_name)

            # File browser widget for choosing file

            file_choose = FileChooserListView(size_hint=(.98, .5), pos_hint={'x': .01, 'y': .22})
            file_choose.bind(on_submit=self.record_filename_click)
            self.img_layout.add_widget(file_choose)

            # Text input for selecting image location

            self.text_input = TextInput(text='File location',
                                        multiline=False, size_hint=(.7, .05), pos_hint={'x': .01, 'y': .14})
            self.text_input.bind(on_text_validate=self.record_filename)
            self.img_layout.add_widget(self.text_input)

        ####################################
        # LOAD IMAGES FROM TXT FILE IN DIR #
        ####################################

        if load_type == 'text':

            # File browser widget for choosing file

            file_choose = FileChooserListView(size_hint=(.98, .5), pos_hint={'x': .01, 'y': .22})
            file_choose.bind(on_submit=self.record_text_file_click)
            self.img_layout.add_widget(file_choose)

            # Text input for selecting image location

            self.text_file_input = TextInput(text='File location',
                                        multiline=False, size_hint=(.7, .05), pos_hint={'x': .01, 'y': .14})
            self.text_file_input.bind(on_text_validate=self.record_text_file)
            self.img_layout.add_widget(self.text_file_input)

        ##############################
        # LOAD IMAGES FROM DIRECTORY #
        ##############################

        if load_type == 'dir':

            file_choose = FileChooserListView(size_hint=(.98, .5), pos_hint={'x': .01, 'y': .22})
            file_choose.bind(on_submit=self.record_dir_click)
            self.img_layout.add_widget(file_choose)

            # Text input for selecting image location

            self.dir_input = TextInput(text='File location',
                                             multiline=False, size_hint=(.7, .05), pos_hint={'x': .01, 'y': .14})
            self.dir_input.bind(on_text_validate=self.record_dir)
            self.img_layout.add_widget(self.dir_input)

    ############################
    # DIRECTORY LOAD FUNCTIONS #
    ############################

    def record_dir_click(self, val, file_name, touch):
        self.load_from_dir(file_name[0])

    def record_dir(self, instance):
        self.load_from_dir(instance.text)

    def load_from_dir(self, file_name):
        if platform.system() == 'Windows':
            file_name_split = file_name.split('\\')
            file_name_split = [s + '\\' for s in file_name_split]
        else:
            file_name_split = file_name.split('/')
            file_name_split = [s + '/' for s in file_name_split]

        dir_name = ''.join(file_name_split[:-1])
        file_list = os.listdir(dir_name)
        file_name_split2 = file_name.split('.')
        file_type = file_name_split2[1]

        # Filter out files of a different file type

        file_list_filtered = []
        for f in file_list:
            if not (f.find(file_type) == -1):
                file_list_filtered.append(dir_name + f)

        self.load_movie([file_list_filtered])

    ############################
    # TEXT FILE LOAD FUNCTIONS #
    ############################

    def record_text_file_click(self, val, file_name, touch):
        self.load_from_textfile(file_name[0])

    def record_text_file(self, instance):
        self.load_from_textfile(instance.text)

    def load_from_textfile(self, file_name):

        if os.path.isfile(file_name):

            if platform.system() == 'Windows':
                file_name_split = file_name.split('\\')
                file_name_split = [s + '\\' for s in file_name_split]
            else:
                file_name_split = file_name.split('/')
                file_name_split = [s + '/' for s in file_name_split]

            dir_name = ''.join(file_name_split[:-1])
            file_list = []
            with open(file_name) as f:
                for line in f:
                    if line[-1] == '\n':
                        line = line[:-1]
                    file_list.append(dir_name + line)

            self.load_movie([file_list])

        else:
            self.error_message.text = '[b][color=000000] Text filename is incorrect [/b][/color]'
            return

    #######################
    # AUTO LOAD FUNCTIONS #
    #######################

    def change_channel(self, channel, obj):

        self.channel = channel
        self.update_file_labels(self.text_input.text)

    def image_pos(self, img_pos, obj):

        self.img_pos = img_pos

    def record_filename(self, instance):

        self.file_names[self.channel*2 + self.img_pos] = instance.text
        self.update_file_labels(instance.text)

    def record_filename_click(self, val, file_name, touch):
        self.file_names[self.channel * 2 + self.img_pos] = file_name[0]
        self.update_file_labels(file_name[0])

    def update_file_labels(self, most_recent_text):

        self.text_input.text = most_recent_text

        if len(self.file_names[self.channel*2]) < 30:
            self.first_img_name.text = '[b][color=000000]' + self.file_names[self.channel*2] + '[/b][/color]'
        else:
            self.first_img_name.text = '[b][color=000000]' + self.file_names[self.channel*2][-29:] + '[/b][/color]'

        if len(self.file_names[self.channel*2+1]) < 30:
            self.last_img_name.text = '[b][color=000000]' + self.file_names[self.channel * 2 + 1] + '[/b][/color]'
        else:
            self.last_img_name.text = '[b][color=000000]' + self.file_names[self.channel * 2 + 1][-29:] + '[/b][/color]'

    def auto_load(self, obj):

        # Autoload all channels where both file names are given output file names to all_file list
        # Handle errors in file loading

        if self.file_names[0] == '' or self.file_names[1] == '':
            self.error_message.text = '[b][color=000000]Select two channel 1 files [/b][/color]'
            return

        if self.file_names[0] == self.file_names[1]:
            self.error_message.text = '[b][color=000000]Select two different files [/b][/color]'
            return

        if not(len(self.file_names[0]) == len(self.file_names[1])):
            self.error_message.text = '[b][color=000000] Names must be of equal length [/b][/color]'
            return

        images = []

        for i in range(self.max_channel):
            if (not self.file_names[i * 2 + 0] == '') and (not self.file_names[i * 2 + 1] == ''):
                if not (self.file_names[i * 2 + 0] ==  self.file_names[i * 2 + 1]):
                    images.append(self.auto_list(self.file_names[i * 2 + 0], self.file_names[i * 2 + 1]))

        flag = self.load_movie(images)

        if flag:

            mx = self.max_channel

            if (not self.file_names[mx * 2 + 0] == '') and (not self.file_names[mx * 2 + 1] == ''):
                if not (self.file_names[mx * 2 + 0] == self.file_names[mx * 2 + 1]):
                    labels = self.auto_list(self.file_names[mx * 2 + 0], self.file_names[mx * 2 + 1])
                    self.load_labels(labels)


    def generate_list_test(self, first_name, dif_loc2):

        if (dif_loc2 + 1) < len(first_name):

            test_digit = first_name[dif_loc2+1]
            flag = True

            if test_digit.isdigit():
                for j in np.arange(int(test_digit) + 1, int(test_digit) + 10):
                    if j < 10:

                        test_digit_temp = str(j)
                        test_name = list(first_name)
                        test_name[dif_loc2+1] = test_digit_temp[0]
                    else:
                        test_digit_temp = str(j)
                        test_name = list(first_name)
                        test_name[dif_loc2] = test_digit_temp[0]
                        test_name[dif_loc2+1] = test_digit_temp[1]

                    if not os.path.isfile(''.join(test_name)):
                        flag = False
            else:
                flag = False

            if flag:

                dif_loc2 += 1
                dif_loc2 = self.generate_list_test(first_name, dif_loc2)

        return dif_loc2

    def auto_list(self, first_name, last_name):

        # Determine whether a difference of one exists somewhere in file names

        fnm = list(first_name)

        l1 = np.asarray([ord(i) for i in list(first_name)])
        l2 = np.asarray([ord(i) for i in list(last_name)])

        dif = l2 - l1

        dif_loc = np.where(dif)[0][0]
        dif_loc2 = np.where(dif)[0][-1]

        # Use recursion to test if last digit is in fact last digit

        dif_loc2 = self.generate_list_test(first_name, dif_loc2)

        nm = dif[dif_loc:dif_loc2 + 1]
        mul = 10 ** len(nm)
        tot = 0
        for x in nm:
            mul //= 10
            tot += x * mul

        pad = len(str(tot))

        start_num = ''
        for i in range(dif_loc, dif_loc + pad):
            start_num += fnm[i]

        start_num = int(start_num)

        file_list = []

        for i in range(tot + 1):
            s = str(start_num + i).zfill(pad)

            for j in range(pad):
                fnm[dif_loc + j] = s[j]

            file_list.append(''.join(fnm))

        return file_list

    ##############################
    # LOAD IMAGES FROM FILE LIST #
    ##############################

    def load_movie(self, image_filenames):

        # Check channels are same length and all files are present

        frames = len(image_filenames[0])

        for all_image_files in image_filenames:

            if not (frames == len(all_image_files)):
                self.error_message.text = '[b][color=000000] Channels not same length [/b][/color]'
                return

            for f in all_image_files:
                if not os.path.isfile(f):
                    self.error_message.text = '[b][color=000000]Missing file: ' + f + '[/b][/color]'
                    return

        # Load images from first channel

        frames = len(image_filenames[0])
        im_test = tifffile.imread(image_filenames[0][0])
        dims = im_test.shape

        images = []
        images_np = []

        for all_image_files in image_filenames:

            all_images = np.zeros((frames, dims[0], dims[1]))

            for i in range(len(all_image_files)):
                im_temp = tifffile.imread(all_image_files[i])
                im_temp = im_temp.astype(float)
                all_images[i, :, :] = im_temp

            images.append(all_images)

            # Transform file names to bytes for storing in hdf5 file
            images_np.append([bytes(image_file, encoding='utf8') for image_file in all_image_files])

        images_np = np.asarray(images_np)

        # Overwrite previous file lists

        for g in self.parent.fov:
            if g == 'image_filenames':
                del self.parent.fov['image_filenames']

        for i in range(len(images)):
            images[i] = images[i]/np.max(images[i].flatten())

        self.parent.images = images
        self.parent.frames = frames
        self.parent.channels = len(images)
        self.parent.dims = dims

        self.parent.fov.create_dataset('image_filenames', data=images_np)
        self.parent.progression[0] = 1
        self.parent.progression[1] = 1
        self.parent.progression_state(2)

        self.error_message.text = '[b][color=000000]Movie Loaded[/b][/color]'
        return True

    def load_labels(self, label_files):

            # Check channels are same length and all files are present
            frames = self.parent.frames
            dims = self.parent.dims

            if not (frames == len(label_files)):
                self.error_message.text = '[b][color=000000] Labels not same length [/b][/color]'
                return

            if not (frames == len(label_files)):
                self.error_message.text = '[b][color=000000] Labels not same length [/b][/color]'
                return

            for f in label_files:
                if not os.path.isfile(f):
                    self.error_message.text = '[b][color=000000]Missing label file: ' + f + '[/b][/color]'
                    return

            im_test = tifffile.imread(label_files[0])
            if not(im_test.shape[0] == dims[0] and im_test.shape[1] == dims[1]):
                self.error_message.text = '[b][color=000000]Label file dimensions different [/b][/color]'
                return

            labels = np.zeros((frames, dims[0], dims[1]))

            for i in range(frames):
                im_temp = tifffile.imread(label_files[i])
                im_temp = im_temp
                labels[i, :, :] = im_temp

            labels = labels.astype(int)
            for g in self.parent.fov:
                if g == 'labels':
                    del self.parent.fov['labels']

            self.parent.params.require_dataset('seg_param', (9,), dtype='f')
            self.parent.fov.create_dataset("labels", data=labels)

            self.parent.progression[2] = 1
            self.parent.progression_state(3)

            self.error_message.text = '[b][color=000000]Movie and Labels Loaded[/b][/color]'

    def remove(self):

        self.ld_layout.clear_widgets()
        self.remove_widget(self.ld_layout)
        self.img_layout.clear_widgets()
        self.remove_widget(self.img_layout)

    def update_size(self, window, width, height):
        self.ld_layout.width = width
        self.ld_layout.height = height
        self.img_layout.width = width
        self.img_layout.height = height
