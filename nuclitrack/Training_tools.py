import numpy as np
import h5py

from kivy.uix.widget import Widget
from kivy.uix.slider import Slider
from kivy.uix.button import Button
from kivy.uix.togglebutton import ToggleButton
from kivy.uix.gridlayout import GridLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.label import Label

from kivy.core.window import Window

from scipy.spatial import distance
from .Image_widget import IndexedDisplay

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
            ypos = (touch.pos[1]-self.pos[1])/self.size[1]
            self.parent.parent.update_training(np.asarray([xpos, ypos]), self.cell_type)

class TrainingUI(Widget):

    def training_frame(self, instance, val):

        self.current_frame = int(val)
        im_temp = self.labels[int(val), :, :]

        mapping = self.features[:, 17].astype(int)
        self.im_disp.update_im(im_temp, mapping)

        inds = self.features[:, 1]
        mask = inds == self.current_frame

        if sum(mask.astype(int)) > 0:
            self.frame_feats = self.features[mask, :]

    def update_training(self, pos, val):

        pos = np.asarray([pos[0] * self.dims[1], pos[1] * self.dims[0]])
        d = distance.cdist(self.frame_feats[:, [2, 3]], [pos])

        sel = self.frame_feats[np.argmin(d), :].copy()

        if min(d) < 50:

            mask = self.training_data[:, 0] == sel[0]

            if np.any(mask):

                ind = np.nonzero(self.training_data[:, 0] == sel[0])
                self.training_data = np.delete(self.training_data, ind, 0)
                print(sel[0])
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

        for g in self.parent.s_param:
            if g == 'training_data':
                del self.parent.s_param['training_data']

        self.parent.s_param.create_dataset("training_data", data=self.training_data)
        self.parent.progression_state(7)

    def tracking_distance(self,instance, val):

        self.parent.track_param[1] = val
        self.txt_dist.text = str(val)

    def max_gap_change(self, instance, val):

        self.parent.track_param[6] = val
        self.txt_gap.text = str(val)

    def mig_cost_change(self, instance, val):

        self.parent.track_param[0] = val
        self.txt_mig.text = str(val)

    def initialize(self, labels, features, frames):

        self.labels = labels
        self.features = features
        self.frames = frames

        self.dims = labels[0, :, :].shape
        self.feature_number = features.shape[1]
        self.current_frame = 0
        self.training_data = np.zeros([1, self.feature_number])

        self.t_layout = FloatLayout(size=(Window.width, Window.height))

        self.im_disp = IndexedDisplay(size_hint=(.75, .7), pos_hint={'x': .2, 'y': .2})
        self.t_layout.add_widget(self.im_disp)

        im_temp = self.labels[0, :, :].astype(float)
        mapping = self.features[:, 17].astype(int)
        self.im_disp.create_im(im_temp, 'Random', mapping)

        inds = self.features[:, 1]
        mask = inds == 0

        if sum(mask.astype(int)) > 0:
            self.frame_feats = self.features[mask, :]

        self.frame_slider = Slider(min=0, max=self.frames - 1, value=1, size_hint=(.3, .1), pos_hint={'x': .2, 'y': .9})
        self.frame_slider.bind(value=self.training_frame)

        self.track_dist = Slider(min=0, max=100, value=50, step=1, size_hint=(.13, .1), pos_hint={'x': .52, 'y': .9})
        self.track_dist.bind(value=self.tracking_distance)
        self.txt_dist = Label(text='50', markup=True, size_hint=(.1, .05),
                                  pos_hint={'x': .5, 'y': .9})

        self.max_gap = Slider(min=0, max=6, value=3, step=1, size_hint=(.13, .1), pos_hint={'x': .67, 'y': .9})
        self.max_gap.bind(value=self.max_gap_change)
        self.txt_gap = Label(text='3', markup=True, size_hint=(.1, .05),
                              pos_hint={'x': .65, 'y': .9})

        self.mig_cost = Slider(min=0, max=0.1, value=0.05, step=0.001, size_hint=(.13, .1), pos_hint={'x': .82, 'y': .9})
        self.mig_cost.bind(value=self.mig_cost_change)
        self.txt_mig = Label(text='0.05', markup=True, size_hint=(.1, .05),
                             pos_hint={'x': .8, 'y': .9})

        self.training_window = TrainingData(size_hint=(.75, .7), pos_hint={'x':.2, 'y':.2})
        layout3 = GridLayout(cols=1, padding=2, size_hint=(.12, .8), pos_hint={'x': .01, 'y': .1})

        btn1 = ToggleButton(text='0 Cell', group='type')
        btn2 = ToggleButton(text='1 Cell', group='type')
        btn3 = ToggleButton(text='2 Cell', group='type')
        btn4 = ToggleButton(text='Mit Cell', group='type')
        btn5 = ToggleButton(text='Mitex Cell', group='type')

        btn6 = Button(text='Save Training')

        btn1.bind(on_press=self.training_window.assign_no_cell)
        btn2.bind(on_press=self.training_window.assign_1_cell)
        btn3.bind(on_press=self.training_window.assign_2_cell)
        btn4.bind(on_press=self.training_window.assign_mit_cell)
        btn5.bind(on_press=self.training_window.assign_mitex_cell)
        btn6.bind(on_press=self.save_training)

        layout3.add_widget(btn1)
        layout3.add_widget(btn2)
        layout3.add_widget(btn3)
        layout3.add_widget(btn4)
        layout3.add_widget(btn5)
        layout3.add_widget(btn6)

        self.layout3 = layout3

        with self.canvas:

            self.add_widget(self.t_layout)
            self.t_layout.add_widget(self.training_window)
            self.t_layout.add_widget(self.frame_slider)
            self.t_layout.add_widget(self.track_dist)
            self.t_layout.add_widget(self.txt_dist)
            self.t_layout.add_widget(self.max_gap)
            self.t_layout.add_widget(self.txt_gap)
            self.t_layout.add_widget(self.mig_cost)
            self.t_layout.add_widget(self.txt_mig)

            self.t_layout.add_widget(self.layout3)

    def remove(self):

        # Remove segmentation ui widgets

        self.layout3.clear_widgets()
        self.t_layout.clear_widgets()
        self.remove_widget(self.t_layout)

    def update_size(self, window, width, height):

        self.t_layout.width = width
        self.t_layout.height = height
