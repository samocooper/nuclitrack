import functools
from multiprocessing import Pool
import numpy as np

import tracking_c_tools
from .Segmentation_tools import SegmentationUI, ViewSegmentation, segment_im
from .Tracking_tools import TrackingUI, RunTracking
from .Training_tools import TrainingUI
from .Loading_tools import FileLoader
from .Image_widget import ImDisplay

from kivy.app import App
from kivy.core.window import Window
from kivy.uix.button import Button
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.uix.slider import Slider
from kivy.uix.togglebutton import ToggleButton
from kivy.uix.widget import Widget
from kivy.uix.progressbar import ProgressBar

from skimage.measure import regionprops
from sklearn.ensemble import RandomForestClassifier
from kivy.clock import Clock
from functools import partial

import time

class UserInterface(Widget):

    def clear_ui(self):

        if self.current_ui == 1:

            self.data_widget.remove()
            self.remove_widget(self.data_widget)

        if self.current_ui == 2:

            self.segment_p.remove()
            self.remove_widget(self.segment_p)

        if self.current_ui == 3:

            self.layout2.clear_widgets()
            self.m_layout.remove_widget(self.layout2)
            self.m_layout.remove_widget(self.seg_message)

        if self.current_ui == 4:

            self.view_segmentation.remove()
            self.remove_widget(self.view_segmentation)

        if self.current_ui == 5:

            self.layout2.clear_widgets()
            self.m_layout.remove_widget(self.layout2)
            self.m_layout.remove_widget(self.feat_message)

        if self.current_ui == 6:

            self.training_p.remove()
            self.remove_widget(self.training_p)

        if self.current_ui == 7:

            self.m_layout.remove_widget(self.track_message)
            self.m_layout.remove_widget(self.track_counter)

        if self.current_ui == 8:

            self.tracking_p.remove()
            self.remove_widget(self.tracking_p)


    def data_ui(self, instance):

        if instance.state == 'down':
            self.clear_ui()
            self.current_ui = 1

            self.data_widget = FileLoader(size_hint=(1., 1.), pos_hint={'x': .01, 'y': .01})
            self.add_widget(self.data_widget)

            self.data_widget.build()
            Window.bind(on_resize=self.data_widget.update_size)

    def segment_ui(self, instance):

        if instance.state == 'down':

            self.clear_ui()
            self.current_ui = 2

            self.segment_p = SegmentationUI(size_hint=(1., 1.), pos_hint={'x': .01, 'y': .01})
            self.add_widget(self.segment_p)

            self.segment_p.initialize(self.frames, len(self.all_channels))
            Window.bind(on_resize=self.segment_p.update_size)

    def segment_movie(self, instance):

        self.clear_ui()
        self.current_ui = 3

        self.seg_message = Label(text='[b][color=000000]Segmenting Images[/b][/color]', markup=True,
                                 size_hint=(.2, .05), pos_hint={'x': .4, 'y': .65})
        self.m_layout.add_widget(self.seg_message)

        self.pb.value = 1000 / self.frames
        self.m_layout.add_widget(self.layout2)
        self.layout2.add_widget(self.pb)

        self.labels = self.fov.require_dataset("labels", (self.frames, self.dims[0], self.dims[1]), dtype='i')
        self.params = self.s_param['seg_param'][:]
        self.frames = self.all_channels[0].shape[0]
        self.label_np = np.zeros(self.all_channels[0].shape)

        self.count_scheduled = 0
        self.count_completed = 0

        self.segment_flag = True

        self.progression_state(4)
        self.progression_state(5)

    def view_segments(self, instance):

        if instance.state == 'down':

            self.clear_ui()
            self.current_ui = 4

            self.view_segmentation = ViewSegmentation(size_hint=(1., 1.), pos_hint={'x': .01, 'y': .01})
            self.add_widget(self.view_segmentation)

            self.view_segmentation.initialize(self.labels, self.frames)


    def update_bar(self, dt):
        self.pb.value += 1000/self.frames

    def segment_im(self, frame, dt):

        self.labels[frame, :, :] = segment_im(self.params, self.all_channels[0][frame, :, :])


    def frame_features(self, frame, dt):

        img_label = self.labels[frame, :, :].copy()
        features_temp = []

        for j in range(self.channel_num):
            features_temp.append(regionprops(img_label, self.all_channels[j][frame, :, :]))

        feature_mat = np.zeros((len(features_temp[0]), self.feature_num))
        img_new = np.zeros(img_label.shape)

        for j in range(len(features_temp[0])):

            cell_temp = features_temp[0][j]

            features_vector = np.zeros(self.feature_num)

            ypos = cell_temp.centroid[0]
            xpos = cell_temp.centroid[1]

            features_vector[0] = self.counter
            features_vector[1] = frame
            features_vector[2] = xpos
            features_vector[3] = ypos
            features_vector[4] = min([ypos, self.dims[0] - ypos, xpos, self.dims[1] - xpos])

            features_vector[5] = cell_temp.area
            features_vector[6] = cell_temp.eccentricity
            features_vector[7] = cell_temp.major_axis_length
            features_vector[8] = cell_temp.perimeter

            mu = cell_temp.mean_intensity
            im_temp = cell_temp.intensity_image.flatten()
            bin_temp = cell_temp.image.flatten()
            im_temp = im_temp[bin_temp]

            features_vector[9] = mu
            features_vector[10] = np.std(im_temp)
            features_vector[11] = np.std(im_temp[im_temp > mu])

            for k in range(1, self.channel_num):
                cell_temp = features_temp[k][j]

                mu = cell_temp.mean_intensity
                im_temp = cell_temp.intensity_image.flatten()
                bin_temp = cell_temp.image.flatten()
                im_temp = im_temp[bin_temp]

                features_vector[20 + (k - 1) * 3 + 0] = mu
                features_vector[20 + (k - 1) * 3 + 1] = np.std(im_temp)
                features_vector[20 + (k - 1) * 3 + 2] = np.std(im_temp[im_temp > mu])

            feature_mat[j, :] = features_vector
            img_new[img_label == cell_temp.label] = self.counter

            self.counter += 1

        self.features = np.vstack((self.features, feature_mat))
        self.labels[frame, :, :] = img_new

    def extract_features(self, instance):

        self.clear_ui()
        self.current_ui = 5

        self.feat_message = Label(text='[b][color=000000]Extracting Features[/b][/color]', markup=True,
                                 size_hint=(.2, .05), pos_hint={'x': .4, 'y': .65})

        self.m_layout.add_widget(self.feat_message)

        self.channel_num = len(self.all_channels)
        self.feature_num = 20 + 3*(self.channel_num-1)
        self.features = np.zeros([1, self.feature_num])
        self.counter = 1

        self.pb.value = 1000 / self.frames
        self.m_layout.add_widget(self.layout2)
        self.layout2.add_widget(self.pb)

        self.feature_flag = True
        self.count_scheduled = 0
        self.count_completed = 0

    def training_ui(self, instance):

        flag = False
        for g in self.fov:
            if g == 'features':
                flag = True

        if instance.state == 'down' and flag == True:

            self.clear_ui()
            self.current_ui = 6
            self.training_p = TrainingUI(size_hint=(1., 1.), pos_hint={'x': .01, 'y': .01})
            self.add_widget(self.training_p)

            self.training_p.initialize(self.labels, self.features, self.frames)
            Window.bind(on_resize=self.training_p.update_size)


    def classify_cells(self, instance):

        self.training_data = self.s_param['training_data'][:, :]
        self.training_data = np.delete(self.training_data, 0, 0)

        clf = RandomForestClassifier(n_estimators=100)

        mask = np.sum(self.training_data[:, 12:17], axis=0) > 0

        if sum(mask) > 1:
            inds = np.where(mask)[0]
            clf = clf.fit(self.training_data[:, 5:10], self.training_data[:, 12+inds].astype(bool))
            probs = clf.predict_proba(self.features[:, 5:10])

            i = 0
            for p in probs:
                if len(p[0]) == 1:
                    p = np.hstack([np.asarray(p), np.zeros((len(p), 1))])
                else:
                    p = np.asarray(p)

                self.features[:, 12 + inds[i]] = p[:, 1]
                i += 1

        if sum(mask) == 1:
            ind = np.where(mask)[0]
            self.features[:, 12 + ind] = 1

        if sum(mask) == 0:
            return

        self.fov['features'][...] = self.features
        self.progression_state(8)

    def finish_tracking(self, instance):

        tracks = self.tracking_instance.finish_optimising()

        # Color labels

        r = int(252 * np.random.rand()) + 3
        ind = tracks[0, 4]

        for j in range(tracks.shape[0]):

            if tracks[j, 4] != ind:
                ind = tracks[j, 4]
                r = int(252 * np.random.rand()) + 3

            self.features[int(tracks[j, 0]), 18] = r

            if tracks[j, 3] > 0:
                self.features[int(tracks[j, 0]), 18] = 5

        self.fov['features'][...] = self.features
        tracks[:, 0] = tracks[:, 0] + 1

        # Delete if tracks already exists otherwise store extracted features

        for g in self.fov:
            if g == 'tracks':
                del self.fov['tracks']

        for g in self.fov:
            if g == 'tracks_stored':
                del self.fov['tracks_stored']

        self.fov.create_dataset("tracks", data=tracks)

        tracks_stored = np.zeros(int(max(tracks[:, 4])))
        self.fov.create_dataset("tracks_stored", data=tracks_stored)

        self.progression_state(9)

    def run_tracking(self, instance):

        self.clear_ui()
        self.current_ui = 7

        self.track_message = Label(text='[b][color=000000] Tracking cells [/b][/color]', markup=True,
                                  size_hint=(.2, .05), pos_hint={'x': .4, 'y': .65})
        self.track_counter = Label(text='[b][color=000000] [/b][/color]', markup=True,
                                  size_hint=(.2, .05), pos_hint={'x': .4, 'y': .6})
        self.m_layout.add_widget(self.track_counter)
        self.m_layout.add_widget(self.track_message)

        self.features[:, 18] = 1
        self.tracking_instance = RunTracking(self.features, self.frames, self.track_param)

        self.tracking_state = 0
        self.min_score = 5
        self.score = self.min_score + 1

        self.optimise_count = 0

        self.tracking_flag = True
        self.add_cell_flag = True

    def tracking_ui(self, instance):

        if instance.state == 'down':

            self.clear_ui()
            self.current_ui = 8

            self.tracking_p = TrackingUI(size_hint=(1., 1.), pos_hint={'x': .01, 'y': .01})
            self.add_widget(self.tracking_p)

            self.tracking_p.initialize(self.all_channels, self.labels, self.frames)
            Window.bind(on_resize=self.tracking_p.update_size)

    def progression_state(self, state):

        if state == 2 and self.progression[2] == 0 and self.progression[0] == 1 and self.progression[1] == 1:

            btn2 = ToggleButton(text='Segment',  group='ui_choice')
            btn2.bind(on_press=self.segment_ui)
            self.layout1.add_widget(btn2)

            self.progression[2] = 1

            for g in self.s_param:
                if g == 'seg_param':
                    state = 3

        if state == 3 and self.progression[3] == 0:

            btn3 = ToggleButton(text='Segment\n  Movie',  group='ui_choice')
            btn3.bind(on_press=self.segment_movie)
            self.layout1.add_widget(btn3)

            self.progression[3] = 1

            for g in self.fov:
                print(g)
                if g == 'labels':

                    # Load labels

                    self.labels = self.fov.require_dataset("labels", (self.frames, self.dims[0], self.dims[1]), dtype='i')

                    state = 4

        if state == 4 and self.progression[4] == 0:

            btn4 = ToggleButton(text='  View\nSegment',  group='ui_choice')
            btn4.bind(on_press=self.view_segments)
            self.layout1.add_widget(btn4)

            self.progression[4] = 1

            btn5 = ToggleButton(text='Extract\nFeatures', group='ui_choice')
            btn5.bind(on_press=self.extract_features)
            self.layout1.add_widget(btn5)

            self.progression[5] = 1

            for g in self.fov:
                if g == 'features':
                    self.features = self.fov['features'][...]
                    state = 6

        if state == 6 and self.progression[6] == 0:

            btn6 = ToggleButton(text='Training\n  Data', group='ui_choice')
            btn6.bind(on_press=self.training_ui)
            self.layout1.add_widget(btn6)

            self.progression[6] = 1

            for g in self.s_param:
                if g == 'training_data':
                    state = 7

        if state == 7 and self.progression[7] == 0:

            btn7 = ToggleButton(text='Classify\n  Cells', group='ui_choice')
            btn7.bind(on_press=self.classify_cells)
            self.layout1.add_widget(btn7)

            self.progression[7] = 1

            for g in self.fov:
                if g == 'features':

                    cl = self.fov['features'][:, 12:15]

                    if sum(cl.flatten()) > 0:

                        state = 8

        if state == 8 and self.progression[8] == 0:
            btn8 = ToggleButton(text='Track\n  Cells', group='ui_choice')
            btn8.bind(on_press=self.run_tracking)
            self.layout1.add_widget(btn8)

            self.progression[8] = 1

            for g in self.fov:
                if g == 'tracks':
                    state = 9

        if state == 9 and self.progression[9] == 0:
            btn9 = ToggleButton(text='View\n  Tracks', group='ui_choice')
            btn9.bind(on_press=self.tracking_ui)
            self.layout1.add_widget(btn9)

            self.progression[9] = 1

    def save_features(self, instance):

        self.features[1:, 17:19] = 1
        self.features = self.features[np.argsort(self.features[:, 0]), :]

        # Delete if features already exists otherwise store extracted features as number of segments may change

        for g in self.fov:
            if g == 'features':
                del self.fov['features']

        self.fov.create_dataset("features", data=self.features)
        self.progression_state(6)

    def add_tracks(self, instance):

        if self.score > self.min_score:

            self.score = self.tracking_instance.add_cell(self.min_score)
            self.add_cell_flag = True

        else:
            self.tracking_state = 1
            Clock.schedule_once(partial(self.update_message, 1), 0)
            self.add_cell_flag = True

    def optimise_tracks(self, dt):

        if self.optimise_count < self.tracking_instance.get_max():

            self.tracking_instance.optimise(self.optimise_count)
            self.optimise_count += 1
            self.add_cell_flag = True

        else:
            self.optimise_count = 0
            Clock.schedule_once(partial(self.update_message, self.tracking_state + 1), 0)
            self.add_cell_flag = True
            self.tracking_flag = False


    def do_work(self, dt):
        self.canvas.ask_update()

        if self.segment_flag:

            Clock.schedule_once(self.update_bar, 0)
            Clock.schedule_once(partial(self.segment_im, self.count_scheduled), 0)
            self.count_scheduled += 1

            if self.count_scheduled == self.frames:
                self.segment_flag = False
                self.seg_message.text = '[b][color=000000]Images Segmented[/b][/color]'

        if self.feature_flag:

            Clock.schedule_once(self.update_bar, 0)
            Clock.schedule_once(partial(self.frame_features, self.count_scheduled), 0)
            self.count_scheduled += 1

            if self.count_scheduled == self.frames:

                self.feat_message.text = '[b][color=000000]Features Extracted[/b][/color]'
                self.feature_flag = False

                Clock.schedule_once(self.save_features, 0)

        if self.tracking_flag:

            if self.tracking_state == 0:

                if self.add_cell_flag:
                    self.add_cell_flag = False
                    Clock.schedule_once(self.add_tracks, 0)
                    Clock.schedule_once(partial(self.update_count, 1), 0)

            if 0 < self.tracking_state <= self.iterations:
                if self.add_cell_flag:
                    self.add_cell_flag = False
                    Clock.schedule_once(partial(self.optimise_tracks), 0)
                    Clock.schedule_once(partial(self.update_count, 2), 0)

            if self.tracking_state > self.iterations:
                Clock.schedule_once(self.finish_tracking, 0)
                Clock.schedule_once(partial(self.update_message, -1), 0)
                self.tracking_flag = False

    def update_count(self, instance, dt):
        if instance == 1:
            self.track_counter.text = '[b][color=000000]' + str(self.tracking_instance.get_count()) + '[/b][/color]'
        if instance == 2:
            self.track_counter.text = '[b][color=000000]' + str(self.optimise_count) + '[/b][/color]'


    def update_message(self, instance, dt):
        if instance >= 0:
            self.track_message.text = '[b][color=000000]Optimising Tracks Sweep ' + str(instance) + ' [/b][/color]'
            self.tracking_state = instance
            self.tracking_flag = True

        else:
            self.track_message.text = '[b][color=000000]Tracking Completed [/b][/color]'

    def initialize(self):

        self.current_ui = 0
        self.track_param = np.asarray([0.05, 50, 1, 5, 0, 1, 3])
        self.progression = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        self.m_layout = FloatLayout(size=(Window.width, Window.height))
        self.layout1 = GridLayout(rows=1, padding=2, size_hint=(.98, .1), pos_hint={'x': .01, 'y': .01})

        btn1 = ToggleButton(text=' Load \nData', group='ui_choice')
        btn1.bind(on_press=self.data_ui)
        self.layout1.add_widget(btn1)

        # Progress bar widget

        self.layout2 = GridLayout(rows=1, padding=2, size_hint=(.9, .1), pos_hint={'x': .05, 'y': .5})
        self.pb = ProgressBar(max=1000, size_hint=(8., 1.), pos_hint={'x': .1, 'y': .6}, value=0)

        self.segment_flag = False
        self.feature_flag = False
        self.tracking_flag = False
        self.iterations = 2
        Clock.schedule_interval(self.do_work, 0)

        with self.canvas:
            self.add_widget(self.m_layout)
            self.m_layout.add_widget(self.layout1)

    def update_size(self, window, width, height):

        print('hello')
        self.m_layout.width = width
        self.m_layout.height = height


class CellTrackApp(App):
    def build(self):

        ui = UserInterface()
        ui.initialize()

        Window.clearcolor = (.8, .8, .8, 1)
        Window.bind(on_resize=ui.update_size)

        return ui
