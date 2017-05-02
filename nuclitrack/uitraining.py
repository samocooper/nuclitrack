import numpy as np
from scipy.spatial import distance

from kivy.uix.widget import Widget
from kivy.uix.slider import Slider
from kivy.uix.button import Button
from kivy.uix.togglebutton import ToggleButton
from kivy.uix.gridlayout import GridLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.label import Label
from kivy.uix.dropdown import DropDown
from kivy.core.window import Window

from .imagewidget import IndexedDisplay, ImDisplay
from . import classifycells

class ClassifyCells(Widget):
    def __init__(self, features, training_data, **kwargs):
        super().__init__(**kwargs)

        self.layout = FloatLayout(size=(Window.width, Window.height))
        self.features = classifycells.classifycells(features, training_data)

        self.class_label = Label(text='[b][color=000000]Cells Classified[/b][/color]', markup=True,
                                 size_hint=(.2, .05), pos_hint={'x': .4, 'y': .65})
        with self.canvas:
            self.add_widget(self.layout)
            self.layout.add_widget(self.class_label)

    def get(self):

        return self.features


class TrainingData(Widget):
    cell_type = 0

    def assign_no_cell(self, instance):

        if instance.state == 'down':

            self.cell_type = 1
        else:
            self.cell_type = 0

    def assign_1_cell(self, instance):

        if instance.state == 'down':

            self.cell_type = 2
        else:
            self.cell_type = 0

    def assign_2_cell(self, instance):

        if instance.state == 'down':

            self.cell_type = 3
        else:
            self.cell_type = 0

    def assign_mit_cell(self, instance):

        if instance.state == 'down':

            self.cell_type = 4
        else:
            self.cell_type = 0

    def assign_mitex_cell(self, instance):

        if instance.state == 'down':

            self.cell_type = 5
        else:
            self.cell_type = 0

    def on_touch_down(self, touch):
        if self.cell_type != 0:
            xpos = (touch.pos[0]-self.pos[0])/self.size[0]
            ypos = 1 - (touch.pos[1]-self.pos[1])/self.size[1]
            self.parent.parent.update_training(np.asarray([xpos, ypos]), self.cell_type)

class TrainingUI(Widget):

    def __init__(self, images=None, labels=None, features=None, frames=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.seg_channel = images
        self.labels = labels
        self.features = features
        self.frames = frames

        self.dims = labels[0, :, :].shape
        self.feature_number = features.shape[1]
        self.current_frame = 0
        self.training_data = np.zeros([1, self.feature_number])

        self.layout = FloatLayout(size=(Window.width, Window.height))

        im_temp = self.labels[0, :, :].astype(float)
        mapping = self.features[:, 17].astype(int)

        self.im_disp = IndexedDisplay(size_hint=(.65, .65), pos_hint={'x': .015, 'y': .15})
        self.layout.add_widget(self.im_disp)

        self.label_disp = ImDisplay(size_hint=(.32, .3225), pos_hint={'x': .67, 'y': .15})
        self.layout.add_widget(self.label_disp)

        self.mov_disp = ImDisplay(size_hint=(.32, .3225), pos_hint={'x': .67, 'y': .4775})
        self.layout.add_widget(self.mov_disp)

        self.label_disp.create_im(im_temp, 'Random')
        self.im_disp.create_im(im_temp, 'Random', mapping)
        mov_temp = self.seg_channel[0, :, :]
        self.mov_disp.create_im(mov_temp, 'PastelHeat')

        inds = self.features[:, 1]
        mask = inds == 0

        if sum(mask.astype(int)) > 0:
            self.frame_feats = self.features[mask, :]

        self.frame_slider = Slider(min=0, max=self.frames - 1, value=1)
        self.frame_slider.bind(value=self.training_frame)
        self.frame_text = Label(text='[color=000000]Frame: ' + str(0) + '[/color]', markup=True)

        self.track_dist = Slider(min=0, max=100, value=50, step=1)
        self.track_dist.bind(value=self.tracking_distance)
        self.txt_dist = Label(text='[color=000000]Search distance: ' + str(50) + '[/color]', markup=True)

        self.max_gap = Slider(min=0, max=6, value=3, step=1)
        self.max_gap.bind(value=self.max_gap_change)
        self.txt_gap = Label(text='[color=000000]Max time gap: ' + str(3) + '[/color]', markup=True)

        self.mig_cost = Slider(min=0, max=0.1, value=0.05, step=0.001)
        self.mig_cost.bind(value=self.mig_cost_change)
        self.txt_mig = Label(text='[color=000000]Migration cost: ' + str(0.05) + '[/color]', markup=True)

        self.training_window = TrainingData(size_hint=(.65, .65), pos_hint={'x': .015, 'y': .15})
        self.layout3 = GridLayout(rows=2, cols=5, padding=2, size_hint=(.98, .12), pos_hint={'x': .01, 'y': .87})

        # Drop down menu for choosing which type of segment

        self.channel_choice = DropDown()

        btn1 = ToggleButton(text='0 Cell', group='type', size_hint_y=None)
        btn2 = ToggleButton(text='1 Cell', group='type', size_hint_y=None)
        btn3 = ToggleButton(text='2 Cell', group='type', size_hint_y=None)
        btn4 = ToggleButton(text='Mit Cell', group='type', size_hint_y=None)
        btn5 = ToggleButton(text='Mitex Cell', group='type', size_hint_y=None)

        btn6 = Button(text='Save Training')

        btn1.bind(on_press=self.training_window.assign_no_cell)
        btn2.bind(on_press=self.training_window.assign_1_cell)
        btn3.bind(on_press=self.training_window.assign_2_cell)
        btn4.bind(on_press=self.training_window.assign_mit_cell)
        btn5.bind(on_press=self.training_window.assign_mitex_cell)
        btn6.bind(on_press=self.save_training)

        self.channel_choice.add_widget(btn1)
        self.channel_choice.add_widget(btn2)
        self.channel_choice.add_widget(btn3)
        self.channel_choice.add_widget(btn4)
        self.channel_choice.add_widget(btn5)

        self.main_button = Button(text='Training Class')
        self.main_button.bind(on_release=self.channel_choice.open)
        self.channel_choice.bind(on_select=lambda instance, x: setattr(self.main_button, 'text', x))

        self.layout3.add_widget(self.main_button)
        self.layout3.add_widget(self.frame_slider)
        self.layout3.add_widget(self.track_dist)
        self.layout3.add_widget(self.max_gap)
        self.layout3.add_widget(self.mig_cost)
        self.layout3.add_widget(btn6)
        self.layout3.add_widget(self.frame_text)
        self.layout3.add_widget(self.txt_dist)
        self.layout3.add_widget(self.txt_gap)
        self.layout3.add_widget(self.txt_mig)


        with self.canvas:

            self.add_widget(self.layout)
            self.layout.add_widget(self.training_window)
            self.layout.add_widget(self.layout3)

    def training_frame(self, instance, val):

        self.current_frame = int(val)
        im_temp = self.labels[int(val), :, :]

        mapping = self.features[:, 17].astype(int)
        self.im_disp.update_im(im_temp, mapping)
        self.label_disp.update_im(np.mod(im_temp, 64))

        mov_temp = self.seg_channel[int(val), :, :]
        self.mov_disp.update_im(mov_temp)
        inds = self.features[:, 1]
        mask = inds == self.current_frame

        if sum(mask.astype(int)) > 0:
            self.frame_feats = self.features[mask, :]

        self.frame_text.text = '[color=000000]Frame: ' + str(int(val)) + '[/color]'

    def update_training(self, pos, val):

        pos = np.asarray([pos[0] * self.dims[1], pos[1] * self.dims[0]])
        d = distance.cdist(self.frame_feats[:, [2, 3]], [pos])

        sel = self.frame_feats[np.argmin(d), :].copy()

        if min(d) < 50:

            mask = self.training_data[:, 0] == sel[0]

            if np.any(mask):

                ind = np.nonzero(self.training_data[:, 0] == sel[0])
                self.training_data = np.delete(self.training_data, ind, 0)
                self.features[int(sel[0]), 17] = 1.

            else:

                sel[12:17] = 0
                sel[11 + val] = 1
                self.training_data = np.vstack((self.training_data, sel))
                self.features[int(sel[0]), 17] = 1.0 + val

        im_temp = self.labels[self.current_frame, :, :]
        mapping = self.features[:, 17].astype(int)
        self.im_disp.update_im(im_temp, mapping)
        self.canvas.ask_update()

    def save_training(self, instance):

        # Delete if features already exists otherwise store extracted features

        for g in self.parent.params:
            if g == 'training_data':
                del self.parent.params['training_data']

        self.parent.params.create_dataset("training_data", data=self.training_data)
        self.parent.progression_state(7)

    def tracking_distance(self,instance, val):

        self.parent.track_param[1] = val
        self.txt_dist.text = '[color=000000]Search distance: ' + str(val) + '[/color]'

    def max_gap_change(self, instance, val):

        self.parent.track_param[6] = val
        self.txt_gap.text = '[color=000000]Max time gap: ' + str(val) + '[/color]'

    def mig_cost_change(self, instance, val):

        self.parent.track_param[0] = val
        self.txt_mig.text = '[color=000000]Migration cost: ' + str(val) + '[/color]'

    def remove(self):

        # Remove segmentation ui widgets

        self.layout3.clear_widgets()
        self.layout.clear_widgets()
        self.remove_widget(self.layout)

    def update_size(self, window, width, height):

        self.layout.width = width
        self.layout.height = height
