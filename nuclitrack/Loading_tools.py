import numpy as np
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
from kivy.core.clipboard import Clipboard
from kivy.uix.gridlayout import GridLayout

class FileLoader(Widget):

    def build(self):

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
                               size_hint=(.44, .05), pos_hint={'x': .05, 'y': .95})
        self.ld_layout.add_widget(self.label_fov)

        # Input

        self.text_input_fov = TextInput(text='example_data.hdf5', multiline=False,
                                        size_hint=(.44, .05), pos_hint={'x': .05, 'y': .9})
        self.text_input_fov.bind(on_text_validate=partial(self.dir, self.text_input_fov.text, 'fov'))
        self.ld_layout.add_widget(self.text_input_fov)

        # Display loaded file

        self.loaded_fov = Label(text='[b][color=000000] [/b][/color]', markup=True,
                                size_hint=(.44, .05), pos_hint={'x': .05, 'y': .85})
        self.ld_layout.add_widget(self.loaded_fov)

        # Assign/Load HDF5 file for storing tracking and segmentation parameter data.

        # Label
        self.label_param = Label(text='[b][color=000000]Parameter Data File[/b][/color]', markup=True,
                                 size_hint=(.44, .05), pos_hint={'x': .51, 'y': .95})
        self.ld_layout.add_widget(self.label_param)

        # Input
        self.text_input_param = TextInput(text='example_params.hdf5', multiline=False,
                                          size_hint=(.44, .05), pos_hint={'x': .51, 'y': .9})
        self.text_input_param.bind(on_text_validate=partial(self.dir, self.text_input_param.text, 'param'))
        self.ld_layout.add_widget(self.text_input_param)

        # Display loaded file

        self.loaded_param = Label(text='[b][color=000000] [/b][/color]', markup=True,
                                  size_hint=(.44, .05), pos_hint={'x': .51, 'y': .85})
        self.ld_layout.add_widget(self.loaded_param)

    def dir(self, input_text, input_type, obj):

        if input_type == 'fov':

            self.parent.fov = h5py.File(input_text, "a")
            self.loaded_fov.text = '[b][color=000000] File loaded: ' + input_text + '[/b][/color]'
            self.file_loaded[0] = True

        if input_type == 'param':
            self.parent.s_param = h5py.File(input_text, "a")
            self.loaded_param.text = '[b][color=000000] File loaded: ' + input_text + '[/b][/color]'
            self.file_loaded[1] = True

        if self.file_loaded[0] and self.file_loaded[1]:

            # Inform user if data already exists in file

            flag = True

            for g in self.parent.fov:
                if g == 'file_list':
                    self.message = Label(text='[b][color=000000] Data exists in the HDF5 file [/b][/color]',
                                              markup=True, size_hint=(.5, .05), pos_hint={'x': .25, 'y': .55})
                    self.ld_layout.add_widget(self.message)
                    flag = False

            # Give user choice of how to load image series

            if flag:

                self.load_choice = GridLayout(rows=1, padding=2, size_hint=(.9, .05), pos_hint={'x': .05, 'y': .8})

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

                # Create master container for storing file_names

                self.all_files = []

            self.parent.progression[0] = 1
            self.parent.progression_state(2)

    def load_imgs(self, load_type, obj):

        self.img_layout.clear_widgets()

        # Info on file loading

        self.error_message = Label(text='[b][color=000000][/b][/color]', markup=True,
                                   size_hint=(.19, .05), pos_hint={'x': .75, 'y': .14})
        self.img_layout.add_widget(self.error_message)

        #########################################
        # LOAD IMAGES FROM FIRST AND LAST NAMES #
        #########################################

        if load_type == 'file':
            self.max_channel = 3
            self.channel = 0
            self.img_pos = 0

            self.file_names = []
            for i in range(self.max_channel*2):
                self.file_names.append('')

            # Layout for file buttons

            self.series_choice = GridLayout(rows=1, padding=2, size_hint=(.9, .05), pos_hint={'x': .05, 'y': .755})

            # Drop down menu for choosing which channel

            self.channel_choice = DropDown()

            for i in range(self.max_channel):

                channel_btn = ToggleButton(text='Channel ' + str(i+1), group='channel', size_hint_y=None, height=25)
                channel_btn.bind(on_press=partial(self.change_channel, i))
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

            file_choose = FileChooserListView(size_hint=(.9, .5), pos_hint={'x': .05, 'y': .22})
            file_choose.bind(on_submit=self.record_filename_click)
            self.img_layout.add_widget(file_choose)

            # Text input for selecting image location

            self.text_input = TextInput(text='File location',
                                        multiline=False, size_hint=(.65, .05), pos_hint={'x': .05, 'y': .14})
            self.text_input.bind(on_text_validate=self.record_filename)
            self.img_layout.add_widget(self.text_input)


        ####################################
        # LOAD IMAGES FROM TXT FILE IN DIR #
        ####################################

        if load_type == 'text':

            # File browser widget for choosing file

            file_choose = FileChooserListView(size_hint=(.9, .5), pos_hint={'x': .05, 'y': .22})
            file_choose.bind(on_submit=self.record_text_file_click)
            self.img_layout.add_widget(file_choose)

            # Text input for selecting image location

            self.text_file_input = TextInput(text='File location',
                                        multiline=False, size_hint=(.65, .05), pos_hint={'x': .05, 'y': .14})
            self.text_file_input.bind(on_text_validate=self.record_text_file)
            self.img_layout.add_widget(self.text_file_input)

        ##############################
        # LOAD IMAGES FROM DIRECTORY #
        ##############################

        if load_type == 'dir':

            file_choose = FileChooserListView(size_hint=(.9, .5), pos_hint={'x': .05, 'y': .22})
            file_choose.bind(on_submit=self.record_dir_click)
            self.img_layout.add_widget(file_choose)

            # Text input for selecting image location

            self.dir_input = TextInput(text='File location',
                                             multiline=False, size_hint=(.65, .05), pos_hint={'x': .05, 'y': .14})
            self.dir_input.bind(on_text_validate=self.record_dir)
            self.img_layout.add_widget(self.dir_input)

    ############################
    # TEXT FILE LOAD FUNCTIONS #
    ############################

    def record_dir_click(self, val, file_name, touch):
        self.load_from_dir(file_name[0])

    def record_dir(self, instance):
        self.load_from_dir(instance.text)

    def load_from_dir(self, file_name):

        try:
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
            file_list = file_list_filtered

            for f in file_list:
                if not os.path.isfile(f):
                    self.error_message.text = '[b][color=000000]Missing file: ' + f + '[/b][/color]'
                    return

            self.all_files.append(file_list)

        except:
            self.error_message.text = '[b][color=000000] Unknown error loading files [/b][/color]'


    ############################
    # TEXT FILE LOAD FUNCTIONS #
    ############################

    def record_text_file_click(self, val, file_name, touch):
        self.load_from_textfile(file_name[0])

    def record_text_file(self, instance):
        self.load_from_textfile(instance.text)

    def load_from_textfile(self, file_name):

        try:

            if os.path.isfile(file_name):
                file_name_split = file_name.split('/')
                file_name_split = [s + '/' for s in file_name_split]
                dir_name = ''.join(file_name_split[:-1])
                file_list = []
                with open(file_name) as f:
                    for line in f:
                        if line[-1] == '\n':
                            line = line[:-1]
                        file_list.append(dir_name + line)

                for f in file_list:
                    if not os.path.isfile(f):
                        self.error_message.text = '[b][color=000000]Missing file: ' + f + '[/b][/color]'
                        return

                self.all_files.append(file_list)

            else:
                self.error_message.text = '[b][color=000000] Text filename is incorrect [/b][/color]'
                return
        except:
            self.error_message.text = '[b][color=000000] Unknown error loading files [/b][/color]'

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

        if len(self.file_names[self.channel * 2]) < 30:
            self.last_img_name.text = '[b][color=000000]' + self.file_names[self.channel * 2 + 1] + '[/b][/color]'
        else:
            self.last_img_name.text = '[b][color=000000]' + self.file_names[self.channel * 2 + 1][-29:] + '[/b][/color]'

    def auto_load(self, obj):

        # Autoload all channels where both file names are given output file names to all_file list
        # Handle errors in file loading
        try:
            if self.file_names[0] == '' or self.file_names[1] == '':
                self.error_message.text = '[b][color=000000]Select two channel 1 files [/b][/color]'
                return

            if self.file_names[0] == self.file_names[1]:
                self.error_message.text = '[b][color=000000]Select two different files [/b][/color]'
                return

            if not(len(self.file_names[0]) == len(self.file_names[1])):
                self.error_message.text = '[b][color=000000] Names must be of equal length [/b][/color]'
                return

            for i in range(self.max_channel):
                if (not self.file_names[i * 2 + 0] == '') and (not self.file_names[i * 2 + 1] == ''):
                    if not (self.file_names[i * 2 + 0] ==  self.file_names[i * 2 + 1]):
                        self.auto_list(self.file_names[i * 2 + 0], self.file_names[i * 2 + 1])
        except:
            self.error_message.text = '[b][color=000000] Unknown error loading files [/b][/color]'

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

        for f in file_list:
            if not os.path.isfile(f):
                self.error_message.text = '[b][color=000000]Missing file: ' + f + '[/b][/color]'
                return

        self.all_files.append(file_list)

    def param_dir(self, instance):
        l_btn4 = Button(text=' Load Params', size_hint=(.175, .05), pos_hint={'x': .8, 'y': .24})
        l_btn4.bind(on_press=self.load_parameters)
        self.ld_layout.add_widget(l_btn4)
        self.param_text = instance.text
        # Test contents of parameter file and update options available in parent view

    def file_assign(self, val, file_name, touch):

        self.file_name_temp = file_name[0]
        self.parent.text_display.text = '[b][color=000000]File selected[/b][/color]'

    def file_assign_txt1(self, instance):
        self.file_names[0] = [instance.text]

    def file_assign_txt2(self, instance):
        self.file_names[1] = [instance.text]

    def file_assign_txt3(self, instance):
        self.file_names[2] = [instance.text]

    def file_assign_txt4(self, instance):
        self.file_names[3] = [instance.text]

    def file_assign_txt5(self, instance):
        self.file_names[4] = [instance.text]

    def file_assign_txt6(self, instance):
        self.file_names[5] = [instance.text]

    def set1a(self, instance):
        self.file_names[0] = [self.file_name_temp]
        self.text_input1.text = self.file_name_temp
        self.parent.text_display.text = '[b][color=000000]Choose file[/b][/color]'

    def set1b(self, instance):

        self.file_names[1] = [self.file_name_temp]
        self.text_input2.text = self.file_name_temp
        self.parent.text_display.text = '[b][color=000000]Choose file[/b][/color]'

    def set2a(self, instance):

        self.file_names[2] = [self.file_name_temp]
        self.text_input3.text = self.file_name_temp
        self.parent.text_display.text = '[b][color=000000]Choose file[/b][/color]'

    def set2b(self, instance):

        self.file_names[3] = [self.file_name_temp]
        self.text_input4.text = self.file_name_temp
        self.parent.text_display.text = '[b][color=000000]Choose file[/b][/color]'

    def set3a(self, instance):

        self.file_names[4] = [self.file_name_temp]
        self.text_input5.text = self.file_name_temp
        self.parent.text_display.text = '[b][color=000000]Choose file[/b][/color]'

    def set3b(self, instance):

        self.file_names[5] = [self.file_name_temp]
        self.text_input6.text = self.file_name_temp
        self.parent.text_display.text = '[b][color=000000]Choose file[/b][/color]'

    def paste_text1(self, instance):
        txt = str(Clipboard.get('UTF8_STRING'))
        self.text_input1.text = txt[2:-1]

    def paste_text2(self, instance):
        txt = str(Clipboard.get('UTF8_STRING'))
        self.text_input2.text = txt[2:-1]

    def paste_text3(self, instance):
        txt = str(Clipboard.get('UTF8_STRING'))
        self.text_input3.text = txt[2:-1]

    def paste_text4(self, instance):
        txt = str(Clipboard.get('UTF8_STRING'))
        self.text_input4.text = txt[2:-1]

    def paste_text5(self, instance):
        txt = str(Clipboard.get('UTF8_STRING'))
        self.text_input5.text = txt[2:-1]

    def paste_text6(self, instance):
        txt = str(Clipboard.get('UTF8_STRING'))
        self.text_input6.text = txt[2:-1]

    def paste_text_data(self, instance):
        txt = str(Clipboard.get('UTF8_STRING'))
        self.text_input_fov.text = txt[2:-1]

    def paste_text_param(self, instance):
        txt = str(Clipboard.get('UTF8_STRING'))
        self.text_input_fov.text = txt[2:-1]

    def set_file_list(self, instance):

        self.file_list = []


        if self.file_names[2] and self.file_names[3]:
            # Determine whether a difference of one exists somewhere in file names

            fnm = list(self.file_names[2][0])

            l1 = np.asarray([ord(i) for i in list(self.file_names[2][0])])
            l2 = np.asarray([ord(i) for i in list(self.file_names[3][0])])

            dif = l2 - l1

            dif_loc = np.where(dif)[0][0]
            dif_loc2 = np.where(dif)[0][-1]

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

            self.file_list.append(file_list)

        if self.file_names[4] and self.file_names[5]:
            # Determine whether a difference of one exists somewhere in file names

            fnm = list(self.file_names[4][0])

            l1 = np.asarray([ord(i) for i in list(self.file_names[4][0])])
            l2 = np.asarray([ord(i) for i in list(self.file_names[5][0])])

            dif = l2 - l1

            dif_loc = np.where(dif)[0][0]
            dif_loc2 = np.where(dif)[0][-1]

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

            self.file_list.append(file_list)
            self.parent.load_movie(file_list)

        if self.file_names[0] and self.file_names[1]:

            # Delete if file_list already exists otherwise store extracted features

            for g in self.parent.fov:
                if g == 'file_list':
                    del self.parent.fov['file_list']

            n_list = []
            for flist_temp in self.file_list:
                n_list_temp = []
                for f in flist_temp:
                    n_list_temp.append(np.asarray([ord(i) for i in list(f)]))
                n_list.append(n_list_temp)

            self.parent.fov.create_dataset("file_list", data=n_list)
            self.parent.load_movie(self.file_list)

        else:
            self.parent.text_display.text = '[b][color=000000]2 File names \nnot chosen[/b][/color]'

    def load_from_data(self, instance):
        flag = True

        for g in self.parent.fov:
            if g == 'file_list':

                flag = False
                n_list = self.parent.fov['file_list'][...]

                file_list = []

                for n_list_temp in n_list:
                    file_list_temp = []

                    for x in n_list_temp:
                        file_list_temp.append(''.join([chr(i) for i in list(x)]))

                    file_list.append(file_list_temp)

                self.parent.load_movie(file_list)
        if flag:
            self.parent.text_display.text = '[b][color=000000]No file list[/b][/color]'



    def data_dir(self, instance):


        # Test contents of fov file and update options available in parent view

        self.file_list = []

        # Add button to load data from this file

        l_btn3 = Button(text='Load data', size_hint=(.175, .05), pos_hint={'x': .8, 'y': .3})
        l_btn3.bind(on_press=self.load_from_data)
        self.ld_layout.add_widget(l_btn3)

        # Load screen parameters for classification tracking and segmentation

        self.text_input_param = TextInput(text='Screen_parameter_data_file', multiline=False,
                                          size_hint=(.6, .05), pos_hint={'x': .175, 'y': .24})

        self.text_input_param.bind(on_text_validate=self.param_dir)
        self.text_input_param.bind(on_double_tap=self.paste_text_param)
        self.ld_layout.add_widget(self.text_input_param)

        # Text input for image sequence range

        self.text_input1 = TextInput(text='Ch1 1st image (double click to paste)',
                                     multiline=False, size_hint=(.23, .05), pos_hint={'x': .175, 'y': .18})

        self.text_input1.bind(on_text_validate=self.file_assign_txt1)
        self.text_input1.bind(on_double_tap=self.paste_text1)

        self.text_input2 = TextInput(text='Ch1 last image',
                                     multiline=False, size_hint=(.23, .05), pos_hint={'x': .175, 'y': .12})

        self.text_input2.bind(on_text_validate=self.file_assign_txt2)
        self.text_input2.bind(on_double_tap=self.paste_text2)

        # Graphical File selection

        file_choose = FileChooserListView(size_hint=(.8, .7), pos_hint={'x': .175, 'y': .36})
        file_choose.bind(on_submit=self.file_assign)

        # Load movie from selection

        layout_btn = GridLayout(cols=1, padding=2, size_hint=(.12, .8), pos_hint={'x': .01, 'y': .18})

        l_btn1 = Button(text=' Load \nMovie')
        l_btn1.bind(on_press=self.set_file_list)

        # File choice buttons for graphical selection

        ch1a = Button(text='Ch1 1st')
        ch1b = Button(text='Ch1 last')

        ch1a.bind(on_press=self.set1a)
        ch1b.bind(on_press=self.set1b)

        layout_btn.add_widget(ch1a)
        layout_btn.add_widget(ch1b)
        layout_btn.add_widget(l_btn1)

        self.ld_layout.add_widget(layout_btn)
        self.ld_layout.add_widget(file_choose)
        self.ld_layout.add_widget(self.text_input1)
        self.ld_layout.add_widget(self.text_input2)

        self.ld_layout.add_widget(self.text_input3)
        self.ld_layout.add_widget(self.text_input4)
        self.ld_layout.add_widget(self.text_input5)
        self.ld_layout.add_widget(self.text_input6)

    def initialize(self):

        self.file_names = ['', '', '', '', '', '']

        self.ld_layout = FloatLayout(size=(Window.width, Window.height))
        self.add_widget(self.ld_layout)

        # Assign HDF5 file for storing data from the current field of view.

        self.text_input_data = TextInput(text='Field_of_view_data_file', multiline=False, size_hint=(.6, .05),
                                         pos_hint={'x': .175, 'y': .3})

        self.text_input_data.bind(on_text_validate=self.data_dir)
        self.text_input_data.bind(on_double_tap=self.paste_text_data)
        self.ld_layout.add_widget(self.text_input_data)

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
