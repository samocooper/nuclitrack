import os
from functools import partial
from pathlib import Path
import h5py

from kivy.core.window import Window
from kivy.uix.button import Button
from kivy.uix.dropdown import DropDown
from kivy.uix.filechooser import FileChooserListView
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.textinput import TextInput
from kivy.uix.togglebutton import ToggleButton
from kivy.uix.widget import Widget

from ..nuclitrack_tools import loadimages
from ..nuclitrack_guitools import guitools
from ..nuclitrack_tools.movieobj import MovieObj

class LoadingUI(Widget):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Master layout for loading screen

        self.ld_layout = FloatLayout(size=(Window.width, Window.height))
        self.add_widget(self.ld_layout)

        self.img_layout = FloatLayout(size=(Window.width, Window.height))
        self.add_widget(self.img_layout)

        # Initialize values for specifying choice of loading ui

        self.choose_type = 0
        self.prev_choose = 0

        # Widget dictionary for loading layout

        self.ld_widgets = dict()

        # Record whether both data and parameter files are loaded

        self.file_loaded = [False, False]

        # initialize file loading widget to home directory

        home = str(Path.home())
        self.ld_widgets['file_chooser'] = FileChooserListView(path=home,
                                                              size_hint=(.98, .5), pos_hint={'x': .01, 'y': .22})

        self.path_input = TextInput(text='Change Path', multiline=False,
                                    size_hint=(.38, .05), pos_hint={'x': .61, 'y': .14})
        self.path_input.bind(on_text_validate=self.change_path)
        self.img_layout.add_widget(self.path_input)

        # Bind functions for when file is clicked or path changes

        self.ld_widgets['file_chooser'].bind(on_entries_cleared=self.dir_change)
        self.ld_widgets['file_chooser'].bind(on_submit=self.dir_click)

        # HDF5 DATA FILE
        # Label

        self.ld_widgets['fov_label'] = guitools.ntlabel(text='Experimental Data File', style=1,
                                                        size_hint=(.245, .05), pos_hint={'x': .01, 'y': .95})

        # Option to select existing data file by clicking

        self.ld_widgets['select_fov'] = ToggleButton(text='Select Existing',
                                                     size_hint=(.145, .04), pos_hint={'x': .26, 'y': .955})
        self.ld_widgets['select_fov'].bind(on_press=self.toggle_fov)

        # Text input for file selection

        self.ld_widgets['text_input_fov'] = TextInput(text='', multiline=False,
                                                      size_hint=(.4, .05), pos_hint={'x': .01, 'y': .9})
        self.ld_widgets['text_input_fov'].bind(on_text_validate=partial(self.file_name_val, 'fov'))

        # Display loaded file

        self.ld_widgets['loaded_fov'] = guitools.ntlabel(text='', style=1,
                                                         size_hint=(.4, .05), pos_hint={'x': .01, 'y': .85})

        # HDF5 PARAMETER FILE
        # Label

        self.ld_widgets['label_param'] = guitools.ntlabel(text='Parameter Data File', style=1,
                                                          size_hint=(.245, .05), pos_hint={'x': .42, 'y': .95})

        # Option to select existing data file by clicking

        self.ld_widgets['select_param'] = ToggleButton(text='Select Existing',
                                                       size_hint=(.145, .04), pos_hint={'x': .67, 'y': .955})
        self.ld_widgets['select_param'].bind(on_press=self.toggle_param)

        # Text input for file selection

        self.ld_widgets['text_input_param'] = TextInput(text='', multiline=False,
                                                        size_hint=(.4, .05), pos_hint={'x': .42, 'y': .9})
        self.ld_widgets['text_input_param'].bind(on_text_validate=partial(self.file_name_val, 'param'))

        # Display loaded file

        self.ld_widgets['loaded_param'] = guitools.ntlabel(text='', style=1,
                                                           size_hint=(.4, .05), pos_hint={'x': .42, 'y': .85})

        # Reload button

        self.ld_widgets['reload'] = Button(text='Reload Data', size_hint=(.16, .04), pos_hint={'x': .83, 'y': .9})
        self.ld_widgets['reload'].bind(on_release=self.reload)

        [self.ld_layout.add_widget(w) for _, w in self.ld_widgets.items()]

    def change_path(self, obj):
        try:
            self.ld_widgets['file_chooser'].path = obj.text

        except FileNotFoundError:
            guitools.notify_msg('Directory not found')

        except NotADirectoryError:
            guitools.notify_msg('Selection not a directory')

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
        self.parent.master_btns.clear_widgets()
        guitools.add_tbtn(layout=self.parent.master_btns, text='Load Data', group='uis', func=self.parent.loading_ui)

        self.ld_layout.clear_widgets()
        self.img_layout.clear_widgets()

        [self.ld_layout.add_widget(w) for _, w in self.ld_widgets.items()]
        self.choose_type = 0

        self.ld_widgets['file_chooser'].bind(on_entries_cleared=self.dir_change)

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
            self.ld_widgets['select_fov'].state = 'normal'
            self.load_data('fov', file_name[0])

        if self.choose_type == 5:
            self.ld_widgets['select_param'].state = 'normal'
            self.load_data('param', file_name[0])

    def dir_change(self, val):

        self.ld_widgets['text_input_fov'].text = os.path.join(self.ld_widgets['file_chooser'].path, 'example_data.hdf5')
        self.ld_widgets['text_input_param'].text = os.path.join(self.ld_widgets['file_chooser'].path, 'example_params.hdf5')

    def erase_data(self, instance):

        self.reload(instance)

        for g in self.parent.fov:
            del self.parent.fov[g]

        self.load_data('', '')
        self.file_loaded = [False, False]

        guitools.ntchange(label=self.ld_widgets['loaded_fov'], text='', style=1)
        guitools.ntchange(label=self.ld_widgets['loaded_param'], text='', style=1)

    def load_data(self, input_type, input_text):

        # Display loaded files

        if input_type == 'fov':

            try:
                self.parent.fov = h5py.File(input_text, "a")

            except OSError:
                guitools.notify_msg('File could not be created or opened\n'
                                          'Invalid data format or permission may be denied')
                return

            input_text_short = ('...' + input_text[-24:]) if len(input_text) > 25 else input_text
            guitools.ntchange(label=self.ld_widgets['loaded_fov'], text='File Loaded: ' + input_text_short, style=1)

            # Set path for saving csv files in future

            self.parent.csv_file = input_text[:-5] + '.csv'
            self.parent.sel_csv_file = input_text[:-5] + '_sel.csv'

            self.file_loaded[0] = True

        if input_type == 'param':

            try:
                self.parent.params = h5py.File(input_text, "a")

            except OSError:
                guitools.notify_msg('File could not be created, permission may be denied')
                return

            input_text_short = ('...' + input_text[-24:]) if len(input_text) > 25 else input_text
            guitools.ntchange(label=self.ld_widgets['loaded_param'], text='File Loaded: ' + input_text_short, style=1)

            self.file_loaded[1] = True

        ########################
        # FILE LOADING OPTIONS #
        ########################

        if self.file_loaded[0]:

            self.erase_btn = Button(text='Erase FOV', size_hint=(.16, .04), pos_hint={'x': .83, 'y': .95})
            self.erase_btn.bind(on_release=self.erase_data)
            self.ld_layout.add_widget(self.erase_btn)

        if self.file_loaded[0] and self.file_loaded[1]:
            self.ld_widgets['file_chooser'].unbind(on_entries_cleared=self.dir_change)

            # Inform user if data already exists in file

            flag = True

            for g in self.parent.fov:
                if g == 'file_list':

                    file_list = loadimages.loadfilelist(self.parent.fov)
                    self.parent.movie = MovieObj(file_list)

                    if self.parent.movie.loaded:

                        # Load image from file list stored in hdf5 file

                        self.message = guitools.ntlabel(text='Data exists in the HDF5 file', style=1,
                                                        size_hint=(.5, .05), pos_hint={'x': .25, 'y': .55})

                        self.ld_layout.add_widget(self.message)
                        self.ld_layout.remove_widget(self.ld_widgets['file_chooser'])
                        flag = False

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

            self.first_img_name = guitools.ntlabel(text='', style=1,
                                                   size_hint=(.44, .05), pos_hint={'x': .05, 'y': .7})
            self.last_img_name = guitools.ntlabel(text='', style=1,
                                                  size_hint=(.44, .05), pos_hint={'x': .51, 'y': .7})

            self.img_layout.add_widget(self.first_img_name)
            self.img_layout.add_widget(self.last_img_name)

            # File browser widget for choosing file

            self.choose_type = 1

            # Text input for selecting image location

            self.text_input = TextInput(text='File location', multiline=False,
                                        size_hint=(.6, .05), pos_hint={'x': .01, 'y': .14})
            self.text_input.bind(on_text_validate=self.record_filename)
            self.img_layout.add_widget(self.text_input)

        ####################################
        # LOAD IMAGES FROM TXT FILE IN DIR #
        ####################################

        if load_type == 'text':

            # File browser widget for choosing file

            self.choose_type = 2

            # Text input for selecting image location

            self.text_file_input = TextInput(text='File location', multiline=False,
                                             size_hint=(.6, .05), pos_hint={'x': .01, 'y': .14})
            self.text_file_input.bind(on_text_validate=self.record_text_file)
            self.img_layout.add_widget(self.text_file_input)

        ##############################
        # LOAD IMAGES FROM DIRECTORY #
        ##############################

        if load_type == 'dir':

            self.choose_type = 3

            # Text input for selecting image location

            self.dir_input = TextInput(text='File location', multiline=False,
                                       size_hint=(.6, .05), pos_hint={'x': .01, 'y': .14})
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
            guitools.notify_msg('File selected is invalid')
        except FileNotFoundError:
            guitools.notify_msg('File not found')
        except IndexError:
            guitools.notify_msg('No images found')


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
            guitools.notify_msg('File is not a text file')
        except ValueError:
            guitools.notify_msg('File is incorrect format')
        except FileNotFoundError:
            guitools.notify_msg('File not found')
        except IndexError:
            guitools.notify_msg('No images found')

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

        first_name = self.file_names[self.channel*2]
        first_name = ('...' + first_name[-24:]) if len(first_name) > 25 else first_name

        last_name = self.file_names[self.channel*2+1]
        last_name = ('...' + last_name[-24:]) if len(last_name) > 25 else last_name

        guitools.ntchange(label=self.first_img_name, text=first_name, style=1)
        guitools.ntchange(label=self.last_img_name, text=last_name, style=1)

    def auto_load(self, instance):

        # Auto load all channels where both file names are given output file names to all_file list

        # Handle errors in file loading

        if self.file_names[0] == '' or self.file_names[1] == '':
            guitools.notify_msg('Select two files')
            return

        if self.file_names[0] == self.file_names[1]:
            guitools.notify_msg('Select two different files')
            return

        if not(len(self.file_names[0]) == len(self.file_names[1])):
            guitools.notify_msg('Names must be of equal length')
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
            guitools.notify_msg('Invalid time series naming format')
        except IndexError:
            guitools.notify_msg('No images found')

    ##############################
    # LOAD IMAGES FROM FILE LIST #
    ##############################

    def load_movie(self, file_list):

        # Check channels are same length and all files are present

        frames = len(file_list[0])

        for channel in file_list:

            if not (frames == len(channel)):
                guitools.notify_msg('Channels not same length')
                return False

            for f in channel:
                if not os.path.isfile(f):
                    guitools.notify_msg('Missing File: ' + f)
                    return False

        # Load images save list of file names into hdf5 file

        self.parent.movie = MovieObj(file_list)
        loadimages.savefilelist(file_list, self.parent.fov)

        self.parent.frames = frames
        self.parent.channels = len(file_list)
        self.parent.file_list = file_list

        self.parent.progression[0] = 1
        self.parent.progression[1] = 1
        self.parent.progression_state(2)

        guitools.notify_msg('Movie Loaded')

        return True

    def load_labels(self, file_list):

            # Check channels are same length and all files are present

            frames = self.parent.frames

            if not (frames == len(file_list)):
                guitools.notify_msg('Labels not same length')
                return

            for f in file_list:
                if not os.path.isfile(f):
                    guitools.notify_msg('Missing label file: ' + f)
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

    def update_size(self, window, width, height):

        self.ld_layout.width = width
        self.ld_layout.height = height
        self.img_layout.width = width
        self.img_layout.height = height
