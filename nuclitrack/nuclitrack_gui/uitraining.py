import numpy as np
from functools import partial

from kivy.core.window import Window
from kivy.uix.button import Button
from kivy.uix.dropdown import DropDown
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.uix.slider import Slider
from kivy.uix.togglebutton import ToggleButton
from kivy.uix.widget import Widget
from scipy.spatial import distance

from ..nuclitrack_guitools.imagewidget import IndexedDisplay, ImDisplay
from ..nuclitrack_tools import classifycells
from ..nuclitrack_guitools import guitools

class ClassifyCells(Widget):
    def __init__(self, features, training, **kwargs):
        super().__init__(**kwargs)

        self.layout = FloatLayout(size=(Window.width, Window.height))
        self.features = classifycells.classifycells(features, training)

        self.class_label = Label(text='[b][color=000000]Cells Classified[/b][/color]', markup=True,
                                 size_hint=(.2, .05), pos_hint={'x': .4, 'y': .65})
        with self.canvas:
            self.add_widget(self.layout)
            self.layout.add_widget(self.class_label)

    def get(self):

        return self.features

    def update_size(self, window, width, height):

        self.width = width
        self.height = height

class TrainingData(Widget):

    cell_type = 0

    def cell_type(self, val, instance):

        if instance.state == 'down':

            self.cell_type = val + 1
        else:
            self.cell_type = 0

    def on_touch_down(self, touch):
        if self.cell_type != 0:
            xpos = (touch.pos[0]-self.pos[0])/self.size[0]
            ypos = 1 - (touch.pos[1]-self.pos[1])/self.size[1]
            self.parent.parent.update_training(np.asarray([xpos, ypos]), self.cell_type)

class TrainingUI(Widget):

    def __init__(self, movie, labels, features, params, stored,  **kwargs):
        super().__init__(**kwargs)

        self.labels = labels

        self.features = dict()
        self.features['tracking'] = features['tracking']
        self.features['data'] = features['data']

        self.dims = labels[0, :, :].shape
        self.current_frame = 0

        self.training = dict()
        self.training['data'] = np.zeros([1, features['data'].shape[1]])
        self.training['tracking'] = np.zeros([1, features['tracking'].shape[1]])

        mask = np.where(features['tracking'][:, 5] > 1)[0]
        if np.count_nonzero(mask):

            self.training['data'] = np.vstack((self.training['data'], features['data'][mask, :]))
            self.training['tracking'] = np.vstack((self.training['tracking'], features['tracking'][mask, :]))
            for i in range(1, self.training['tracking'].shape[0]):
                val = features['tracking'][mask[i-1], 5]
                self.training['tracking'][i, int(4 + val)] = 1

        self.layout = FloatLayout(size=(Window.width, Window.height))

        im_temp = self.labels[0, :, :].astype(float)
        mapping = self.features['tracking'][:, 5].astype(int)

        self.im_disp = IndexedDisplay(size_hint=(.65, .65), pos_hint={'x': .015, 'y': .15})
        self.layout.add_widget(self.im_disp)

        self.label_disp = ImDisplay(size_hint=(.32, .3225), pos_hint={'x': .67, 'y': .15})
        self.layout.add_widget(self.label_disp)

        self.mov_disp = ImDisplay(size_hint=(.32, .3225), pos_hint={'x': .67, 'y': .4775})
        self.layout.add_widget(self.mov_disp)

        self.label_disp.create_im(im_temp, 'Random')
        self.im_disp.create_im(im_temp, 'Random', mapping)

        self.movie = movie
        self.mov_disp.create_im(self.movie.read_im(0, 0), 'PastelHeat')
        inds = self.features['tracking'][:, 1]

        mask = inds == self.current_frame
        if np.count_nonzero(mask):
            self.frame_inds = np.where(mask)[0]

        self.frame_slider = guitools.FrameSlider(self.movie.frames, self.change_frame,
                                                 size_hint=(.27, .06), pos_hint={'x': .01, 'y': .88})

        self.track_dist = Slider(min=0, max=100, value=float(params[1]), step=1)
        self.track_dist.bind(value=self.tracking_distance)
        self.txt_dist = Label(text='[color=000000]Search Distance: ' + str(float(params[1])) + '[/color]', markup=True)

        self.max_gap = Slider(min=0, max=6, value=float(params[6]), step=1)
        self.max_gap.bind(value=self.max_gap_change)
        self.txt_gap = Label(text='[color=000000]Max Time Gap: ' + str(float(params[6])) + '[/color]', markup=True)

        self.mig_cost = Slider(min=0, max=0.1, value=float(params[0]), step=0.001)
        self.mig_cost.bind(value=self.mig_cost_change)
        self.txt_mig = Label(text='[color=000000]Migration Cost: ' + str(float(params[0])) + '[/color]', markup=True)

        self.training_window = TrainingData(size_hint=(.65, .65), pos_hint={'x': .015, 'y': .15})
        self.layout3 = GridLayout(rows=3, cols=4, padding=2, size_hint=(.7, .13), pos_hint={'x': .29, 'y': .82})

        # Drop down menu for choosing which type of segment

        self.channel_choice = DropDown()

        labels = ['No Cell', '1 Cell', '2 Cells', 'Mitotic', 'Daughter']

        for i in range(5):
            btn = ToggleButton(text=labels[i], group='type', size_hint_y=None)
            btn.bind(on_press=partial(self.training_window.cell_type, i))
            self.channel_choice.add_widget(btn)

        self.main_button = Button(text='Training Class', size_hint=(.27, .04), pos_hint={'x': .01, 'y': .82})
        self.main_button.bind(on_release=self.channel_choice.open)
        self.channel_choice.bind(on_select=lambda instance, x: setattr(self.main_button, 'text', x))

        btn6 = Button(text='Save Training')
        self.data_message = Label(text='[color=000000][/color]', markup=True)
        btn6.bind(on_press=self.save_training)

        self.layout3.add_widget(self.track_dist)
        self.layout3.add_widget(self.max_gap)
        self.layout3.add_widget(self.mig_cost)
        self.layout3.add_widget(btn6)
        self.layout3.add_widget(self.txt_dist)
        self.layout3.add_widget(self.txt_gap)
        self.layout3.add_widget(self.txt_mig)
        self.layout3.add_widget(self.data_message)

        if stored:
            self.data_message.text = '[color=000000]Data Stored[/color]'

        with self.canvas:

            self.add_widget(self.layout)

            self.layout.add_widget(self.main_button)
            self.layout.add_widget(self.training_window)
            self.layout.add_widget(self.layout3)
            self.layout.add_widget(self.frame_slider)

    def change_frame(self, val):

        self.current_frame = val
        im_temp = self.labels[int(val), :, :]

        mapping = self.features['tracking'][:, 5].astype(int)
        self.im_disp.update_im(im_temp, mapping)
        self.label_disp.update_im(np.mod(im_temp, 64))

        self.mov_disp.update_im(self.movie.read_im(0, self.current_frame))

        inds = self.features['tracking'][:, 1]

        mask = inds == self.current_frame
        if np.count_nonzero(mask):
            self.frame_inds = np.where(mask)[0]

    def update_training(self, pos, val):

        try:
            pos = np.asarray([pos[0] * self.dims[1], pos[1] * self.dims[0]])
            d = distance.cdist(self.features['tracking'][self.frame_inds, 2:4], [pos])

            selected_loc = self.frame_inds[np.argmin(d)]
            selected_ind = self.features['tracking'][selected_loc, 0]

            if min(d) < 50:

                mask = self.training['tracking'][:, 0] == selected_ind

                if np.any(mask):

                    ind = np.nonzero(mask)
                    self.training['data'] = np.delete(self.training['data'], ind, 0)
                    self.training['tracking'] = np.delete(self.training['tracking'], ind, 0)
                    self.features['tracking'][int(selected_ind), 5] = 1.

                else:
                    sel_data = self.features['data'][int(selected_ind), :].copy()
                    sel_tracking = self.features['tracking'][int(selected_ind), :].copy()
                    sel_tracking[6:11] = 0
                    sel_tracking[5 + val] = 1

                    self.training['data'] = np.vstack((self.training['data'], sel_data))
                    self.training['tracking'] = np.vstack((self.training['tracking'], sel_tracking))

                    self.features['tracking'][int(selected_ind), 5] = 1.0 + val

            im_temp = self.labels[self.current_frame, :, :]
            mapping = self.features['tracking'][:, 5].astype(int)
            self.im_disp.update_im(im_temp, mapping)
            self.canvas.ask_update()

        except TypeError:
            pass

    def save_training(self, instance):
        if self.training['data'].shape[0] > 1:

            # Delete if features already exists otherwise store extracted features

            if 'training' in self.parent.params:
                del self.parent.params['training']

            self.training_hdf5 = self.parent.params.create_group('training')
            self.training_hdf5.create_dataset("data", data=self.training['data'])
            self.training_hdf5.create_dataset("tracking", data=self.training['tracking'])

            self.data_message.text ='[color=000000]Data Stored[/color]'
            self.parent.progression_state(7)

    def tracking_distance(self,instance, val):

        self.parent.params['track_param'][1] = val
        self.txt_dist.text = '[color=000000]Search Distance: ' + str(val) + '[/color]'

    def max_gap_change(self, instance, val):

        self.parent.params['track_param'][6] = val
        self.txt_gap.text = '[color=000000]Max Time Gap: ' + str(val) + '[/color]'

    def mig_cost_change(self, instance, val):

        self.parent.params['track_param'][0] = val
        self.txt_mig.text = '[color=000000]Migration Cost: ' + str(val) + '[/color]'

    def remove(self):

        # Remove segmentation ui widgets

        self.layout3.clear_widgets()
        self.layout.clear_widgets()
        self.remove_widget(self.layout)

    def update_size(self, window, width, height):

        self.layout.width = width
        self.layout.height = height
