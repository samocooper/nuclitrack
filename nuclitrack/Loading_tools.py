import numpy as np
import h5py

from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.textinput import TextInput
from kivy.core.window import Window
from kivy.uix.filechooser import FileChooserListView
from kivy.core.clipboard import Clipboard
from kivy.uix.gridlayout import GridLayout

class FileLoader(Widget):

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
        self.text_input_data.text = txt[2:-1]

    def paste_text_param(self, instance):
        txt = str(Clipboard.get('UTF8_STRING'))
        self.text_input_data.text = txt[2:-1]

    def set_file_list(self, instance):

        self.file_list = []

        if self.file_names[0] and self.file_names[1]:
            # Determine whether a difference of one exists somewhere in file names

            fnm = list(self.file_names[0][0])

            l1 = np.asarray([ord(i) for i in list(self.file_names[0][0])])
            l2 = np.asarray([ord(i) for i in list(self.file_names[1][0])])

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

    def load_parameters(self, instance):

        self.parent.s_param = h5py.File(self.param_text, "a")
        self.parent.progression[0] = 1
        self.parent.progression_state(2)

    def param_dir(self, instance):

        l_btn4 = Button(text=' Load Params', size_hint=(.175, .05), pos_hint={'x': .8, 'y': .24})
        l_btn4.bind(on_press=self.load_parameters)
        self.ld_layout.add_widget(l_btn4)
        self.param_text = instance.text
        # Test contents of parameter file and update options available in parent view

    def data_dir(self, instance):

        self.file_name_temp = ''
        self.parent.fov = h5py.File(instance.text, "a")

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

        # Channel 2

        self.text_input3 = TextInput(text='Ch2 1st image (optional)',
                                     multiline=False, size_hint=(.23, .05), pos_hint={'x': .435, 'y': .18})

        self.text_input3.bind(on_text_validate=self.file_assign_txt3)
        self.text_input3.bind(on_double_tap=self.paste_text3)

        self.text_input4 = TextInput(text='Ch2 last image',
                                     multiline=False, size_hint=(.23, .05), pos_hint={'x': .435, 'y': .12})

        self.text_input4.bind(on_text_validate=self.file_assign_txt4)
        self.text_input4.bind(on_double_tap=self.paste_text4)

        # Channel 3

        self.text_input5 = TextInput(text='Ch3 1st image (optional)',
                                     multiline=False, size_hint=(.23, .05), pos_hint={'x': .695, 'y': .18})

        self.text_input5.bind(on_text_validate=self.file_assign_txt5)
        self.text_input5.bind(on_double_tap=self.paste_text5)

        self.text_input6 = TextInput(text='Ch3 last image',
                                     multiline=False, size_hint=(.23, .05), pos_hint={'x': .695, 'y': .12})

        self.text_input6.bind(on_text_validate=self.file_assign_txt6)
        self.text_input6.bind(on_double_tap=self.paste_text6)

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
        ch2a = Button(text='Ch2 1st')
        ch2b = Button(text='Ch2 last')
        ch3a = Button(text='Ch2 1st')
        ch3b = Button(text='Ch2 last')

        ch1a.bind(on_press=self.set1a)
        ch1b.bind(on_press=self.set1b)
        ch2a.bind(on_press=self.set2a)
        ch2b.bind(on_press=self.set2b)
        ch3a.bind(on_press=self.set3a)
        ch3b.bind(on_press=self.set3b)

        layout_btn.add_widget(ch1a)
        layout_btn.add_widget(ch1b)
        layout_btn.add_widget(ch2a)
        layout_btn.add_widget(ch2b)
        layout_btn.add_widget(ch3a)
        layout_btn.add_widget(ch3b)

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

    def update_size(self, window, width, height):
        self.ld_layout.width = width
        self.ld_layout.height = height
