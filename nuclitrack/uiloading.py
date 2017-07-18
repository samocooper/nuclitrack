import h5py
import os
from functools import partial

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
from . import loadimages


class LoadingUI(Widget):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Master layout for loading screen

        self.ld_layout = FloatLayout(size=(Window.width, Window.height))
        self.add_widget(self.ld_layout)

        self.img_layout = FloatLayout(size=(Window.width, Window.height))
        self.add_widget(self.img_layout)
        self.choose_type = 0
        # Assign/Load HDF5 file for storing data from the current field of view.

        # Record whether both files are loaded
        self.file_loaded = [False, False]

        self.file_choose = FileChooserListView(size_hint=(.98, .5), pos_hint={'x': .01, 'y': .22})
        self.ld_layout.add_widget(self.file_choose)
        self.file_choose.bind(on_entries_cleared=self.dir_change)
        self.file_choose.bind(on_submit=self.dir_click)

        # Label
        self.label_fov = Label(text='[b][color=000000]Experimental Data File[/b][/color]', markup=True,
                               size_hint=(.245, .05), pos_hint={'x': .01, 'y': .95})
        self.ld_layout.add_widget(self.label_fov)

        self.select_fov = ToggleButton(text='Select Existing', size_hint=(.145, .04), pos_hint={'x': .26, 'y': .955})
        self.ld_layout.add_widget(self.select_fov)
        self.select_fov.bind(on_press=self.toggle_fov)

        # Input
        self.text_input_fov = TextInput(text='', multiline=False,
                                        size_hint=(.4, .05), pos_hint={'x': .01, 'y': .9})
        self.text_input_fov.bind(on_text_validate=partial(self.file_name_val, 'fov'))
        self.ld_layout.add_widget(self.text_input_fov)

        # Display loaded file

        self.loaded_fov = Label(text='[b][color=000000] [/b][/color]', markup=True,
                                size_hint=(.4, .05), pos_hint={'x': .01, 'y': .85})
        self.ld_layout.add_widget(self.loaded_fov)

        # Info on file loading

        self.ui_message = Label(text='[b][color=000000][/b][/color]', markup=True,
                                size_hint=(.19, .05), pos_hint={'x': .75, 'y': .14})
        self.ld_layout.add_widget(self.ui_message)

        # Assign/Load HDF5 file for storing tracking and segmentation parameter data.

        # Label
        self.label_param = Label(text='[b][color=000000]Parameter Data File[/b][/color]', markup=True,
                                 size_hint=(.245, .05), pos_hint={'x': .42, 'y': .95})
        self.ld_layout.add_widget(self.label_param)

        self.select_param = ToggleButton(text='Select Existing', size_hint=(.145, .04), pos_hint={'x': .67, 'y': .955})
        self.ld_layout.add_widget(self.select_param)
        self.select_param.bind(on_press=self.toggle_param)

        # Input
        self.text_input_param = TextInput(text='', multiline=False,
                                          size_hint=(.4, .05), pos_hint={'x': .42, 'y': .9})
        self.text_input_param.bind(on_text_validate=partial(self.file_name_val, 'param'))
        self.ld_layout.add_widget(self.text_input_param)

        # Display loaded file

        self.loaded_param = Label(text='[b][color=000000] [/b][/color]', markup=True,
                                  size_hint=(.4, .05), pos_hint={'x': .42, 'y': .85})

        self.ld_layout.add_widget(self.loaded_param)

        # Reload button

        self.reload_btn = Button(text='Reload Data', size_hint=(.16, .04), pos_hint={'x': .83, 'y': .9})
        self.reload_btn.bind(on_release=self.reload)
        self.ld_layout.add_widget(self.reload_btn)

    def toggle_fov(self, instance):

        if instance.state == 'down':
            self.prev_choose = self.choose_type
            self.choose_type = 4
        else:
            self.choose_type = self.prev_choose

    def toggle_param(self, instance):

        if instance.state == 'down':
            self.prev_choose = self.choose_type
            self.choose_type = 5
        else:
            self.choose_type = self.prev_choose

    def reload(self, instance, erase_flag=False):

        self.parent.progression = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.parent.layout1.clear_widgets()
        self.parent.layout1.add_widget(self.parent.btn1)

        self.ld_layout.clear_widgets()
        self.img_layout.clear_widgets()

        self.ld_layout.add_widget(self.label_fov)
        self.ld_layout.add_widget(self.text_input_fov)
        self.ld_layout.add_widget(self.select_fov)
        self.ld_layout.add_widget(self.loaded_fov)
        self.ld_layout.add_widget(self.ui_message)
        self.ld_layout.add_widget(self.label_param)
        self.ld_layout.add_widget(self.select_param)
        self.ld_layout.add_widget(self.text_input_param)
        self.ld_layout.add_widget(self.loaded_param)
        self.ld_layout.add_widget(self.reload_btn)
        self.ld_layout.add_widget(self.file_choose)

        self.choose_type = 0
        self.file_choose.bind(on_entries_cleared=self.dir_change)

        if not erase_flag:

            self.file_loaded = [False, False]
            self.loaded_fov.text ='[b][color=000000] [/b][/color]'
            self.loaded_param.text = '[b][color=000000] [/b][/color]'

    def file_name_val(self, input_type, obj):
        self.load_data(input_type, obj.text)

    def dir_click(self, val, file_name, touch):

        if self.choose_type == 1:
            self.file_names[self.channel * 2 + self.img_pos] = file_name[0]
            self.update_file_labels(file_name[0])

        if self.choose_type == 2:
            self.load_from_textfile(file_name[0])

        if self.choose_type == 3:
            self.load_from_dir(file_name[0])

        if self.choose_type == 4:
            self.select_fov.state = 'normal'
            self.load_data('fov', file_name[0])

        if self.choose_type == 5:
            self.select_param.state = 'normal'
            self.load_data('param', file_name[0])

    def dir_change(self, val):

        self.text_input_fov.text = os.path.join(self.file_choose.path, 'example_data.hdf5')
        self.text_input_param.text = os.path.join(self.file_choose.path, 'example_params.hdf5')

    def erase_data(self, instance):

        self.reload(instance, erase_flag=True)

        for g in self.parent.fov:
            del self.parent.fov[g]

        self.load_data('', '')

    def load_data(self, input_type, input_text):

        # Display loaded files

        if input_type == 'fov':

            try:
                self.parent.fov = h5py.File(input_text, "a")

            except OSError:
                self.parent.error_message('File could not be created or opened\n'
                                          'Invalid data format or permission may be denied')
                return

            if len(input_text) < 20:
                self.loaded_fov.text = '[b][color=000000] File loaded: ' + input_text + '[/b][/color]'
            else:
                self.loaded_fov.text = '[b][color=000000] File loaded: ' + input_text[-30:] + '[/b][/color]'

            # Set path for saving csv files in future

            self.parent.csv_file = input_text[:-5] + '.csv'
            self.parent.sel_csv_file = input_text[:-5] + '_sel.csv'

            self.file_loaded[0] = True

        if input_type == 'param':

            try:
                self.parent.params = h5py.File(input_text, "a")

            except OSError:
                self.parent.error_message('File could not be created, permission may be denied')
                return

            if len(input_text) < 20:
                self.loaded_param.text = '[b][color=000000] File Loaded: ' + input_text + '[/b][/color]'
            else:
                self.loaded_param.text = '[b][color=000000] File Loaded: ' + input_text[-30:] + '[/b][/color]'

            self.file_loaded[1] = True

        ########################
        # FILE LOADING OPTIONS #
        ########################

        if self.file_loaded[0]:

            self.erase_btn = Button(text='Erase FOV', size_hint=(.16, .04), pos_hint={'x': .83, 'y': .95})
            self.erase_btn.bind(on_release=self.erase_data)
            self.ld_layout.add_widget(self.erase_btn)

        if self.file_loaded[0] and self.file_loaded[1]:

            self.file_choose.unbind(on_entries_cleared=self.dir_change)

            # Inform user if data already exists in file

            flag = True

            for g in self.parent.fov:
                if g == 'file_list':

                    self.message = Label(text='[b][color=000000] Data exists in the HDF5 file [/b][/color]',
                                              markup=True, size_hint=(.5, .05), pos_hint={'x': .25, 'y': .55})
                    self.ld_layout.add_widget(self.message)
                    self.ld_layout.remove_widget(self.file_choose)
                    flag = False

                    file_list = loadimages.loadfilelist(self.parent.fov)

                    self.parent.dims, self.parent.min_vals, self.parent.max_vals, = loadimages.loadimages(file_list)
                    self.parent.file_list = file_list
                    self.parent.frames = len(file_list[0])
                    self.parent.channels = len(file_list)

                    self.parent.progression[0] = 1
                    self.parent.progression[1] = 1
                    self.parent.progression_state(2)

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

            load_image = Button(text='Load Images')
            load_image.bind(on_press=self.auto_load)
            self.series_choice.add_widget(load_image)

            self.img_layout.add_widget(self.series_choice)

            # Labels for informing users which images have been loaded

            self.first_img_name = Label(text='[b][color=000000] [/b][/color]',
                                 markup=True, size_hint=(.44, .05), pos_hint={'x': .05, 'y': .7})
            self.last_img_name = Label(text='[b][color=000000] [/b][/color]',
                                 markup=True, size_hint=(.44, .05), pos_hint={'x': .51, 'y': .7})

            self.img_layout.add_widget(self.first_img_name)
            self.img_layout.add_widget(self.last_img_name)

            # File browser widget for choosing file

            self.choose_type = 1

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

            self.choose_type = 2

            # Text input for selecting image location

            self.text_file_input = TextInput(text='File location',
                                        multiline=False, size_hint=(.7, .05), pos_hint={'x': .01, 'y': .14})
            self.text_file_input.bind(on_text_validate=self.record_text_file)
            self.img_layout.add_widget(self.text_file_input)

        ##############################
        # LOAD IMAGES FROM DIRECTORY #
        ##############################

        if load_type == 'dir':

            self.choose_type = 3

            # Text input for selecting image location

            self.dir_input = TextInput(text='File location',
                                             multiline=False, size_hint=(.7, .05), pos_hint={'x': .01, 'y': .14})
            self.dir_input.bind(on_text_validate=self.record_dir)
            self.img_layout.add_widget(self.dir_input)

    ############################
    # DIRECTORY LOAD FUNCTIONS #
    ############################

    def record_dir(self, instance):
        self.load_from_dir(instance.text)

    def load_from_dir(self, dir_name):
        try:
            file_list = loadimages.filelistfromdir(dir_name)
            self.load_movie(file_list)

        except ValueError:
            self.parent.error_message('File selected is invalid')
        except FileNotFoundError:
            self.parent.error_message('File not found')
        except IndexError:
            self.parent.error_message('No images found')


    ############################
    # TEXT FILE LOAD FUNCTIONS #
    ############################

    def record_text_file(self, instance):

        self.load_from_textfile(instance.text)

    def load_from_textfile(self, text_file):

        try:

            file_list, label_list = loadimages.filelistfromtext(text_file)
            flag = self.load_movie(file_list)
            if len(label_list) > 1 and flag:
                self.load_labels(label_list)

        except UnicodeDecodeError:
            self.parent.error_message('File is not a text file')
        except ValueError:
            self.parent.error_message('File is incorrect format')
        except FileNotFoundError:
            self.parent.error_message('File not found')
        except IndexError:
            self.parent.error_message('No images found')

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

    def auto_load(self, instance):

        # Autoload all channels where both file names are given output file names to all_file list
        # Handle errors in file loading

        if self.file_names[0] == '' or self.file_names[1] == '':
            self.parent.error_message('Select two files')
            return

        if self.file_names[0] == self.file_names[1]:
            self.parent.error_message('Select two different files')
            return

        if not(len(self.file_names[0]) == len(self.file_names[1])):
            self.parent.error_message('Names must be of equal length')
            return

        images = []

        try:

            for i in range(self.max_channel):
                if (not self.file_names[i * 2 + 0] == '') and (not self.file_names[i * 2 + 1] == ''):
                    if not (self.file_names[i * 2 + 0] == self.file_names[i * 2 + 1]):
                        images.append(loadimages.autofilelist(self.file_names[i * 2 + 0], self.file_names[i * 2 + 1]))

            flag = self.load_movie(images)

            if flag:

                mx = self.max_channel

                if (not self.file_names[mx * 2 + 0] == '') and (not self.file_names[mx * 2 + 1] == ''):
                    if not (self.file_names[mx * 2 + 0] == self.file_names[mx * 2 + 1]):

                        labels = loadimages.autofilelist(self.file_names[mx * 2 + 0], self.file_names[mx * 2 + 1])
                        self.load_labels(labels)

        except ValueError:
            self.parent.error_message('Invalid time series naming format')
        except IndexError:
            self.parent.error_message('No images found')

    ##############################
    # LOAD IMAGES FROM FILE LIST #
    ##############################

    def load_movie(self, file_list):

        # Check channels are same length and all files are present

        frames = len(file_list[0])

        for channel in file_list:

            if not (frames == len(channel)):
                self.parent.error_message('Channels not same length')
                return False

            for f in channel:
                if not os.path.isfile(f):
                    self.parent.error_message('Missing File: ' + f)
                    return False

        # Load images save list of file names into hdf5 file

        self.parent.dims, self.parent.min_vals, self.parent.max_vals, = loadimages.loadimages(file_list)

        loadimages.savefilelist(file_list, self.parent.fov)

        self.parent.frames = frames
        self.parent.channels = len(file_list)
        self.parent.file_list = file_list

        self.parent.progression[0] = 1
        self.parent.progression[1] = 1
        self.parent.progression_state(2)

        self.ui_message.text = '[b][color=000000]Movie Loaded[/b][/color]'

        return True

    def load_labels(self, file_list):

            # Check channels are same length and all files are present

            frames = self.parent.frames

            if not (frames == len(file_list)):
                self.parent.error_message('Labels not same length')
                return

            for f in file_list:
                if not os.path.isfile(f):
                    self.parent.error_message('Missing label file: ' + f)
                    return

            # Load and save label images

            labels = loadimages.loadlabels(file_list)

            for g in self.parent.fov:
                if g == 'labels':
                    del self.parent.fov['labels']

            self.parent.fov.create_dataset("labels", data=labels)

            # initialise segmentation parameters with dummy variables for state progression

            self.parent.params.require_dataset('seg_param', (11,), dtype='f')

            self.parent.progression[2] = 1
            self.parent.progression_state(3)
            self.ui_message.text = '[b][color=000000]Movie and Labels Loaded[/b][/color]'

    def update_size(self, window, width, height):

        self.ld_layout.width = width
        self.ld_layout.height = height
        self.img_layout.width = width
        self.img_layout.height = height
