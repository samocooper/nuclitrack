import functools
from multiprocessing import Pool
import numpy as np

import tracking_c_tools
from .Segmentation_tools import SegmentationUI, segment_im
from .Tracking_tools import TrackingUI
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

            self.m_layout.remove_widget(self.im_disp)
            self.m_layout.remove_widget(self.sframe)

        if self.current_ui == 5:

            self.layout2.clear_widgets()
            self.m_layout.remove_widget(self.layout2)
            self.m_layout.remove_widget(self.feat_message)

        if self.current_ui == 6:

            self.training_p.remove()
            self.remove_widget(self.training_p)


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

            im_temp = self.labels[0, :, :]

            self.im_disp = ImDisplay(size_hint=(.75, .7), pos_hint={'x': .2, 'y': .2})
            self.im_disp.create_im(im_temp, 'Random')
            self.m_layout.add_widget(self.im_disp)

            self.sframe = Slider(min=0, max=self.frames - 1, value=1, size_hint=(.3, .1),
                                 pos_hint={'x': .2, 'y': .9})
            self.sframe.bind(value=self.segment_frame)
            self.m_layout.add_widget(self.sframe)

    def update_bar(self, dt):
        self.pb.value += 1000/self.frames

    def segment_im(self, frame, dt):

        self.labels[frame, :, :] = segment_im(self.params, self.all_channels[0][frame, :, :])

    def segment_frame(self, instance, val):

        im_temp = self.labels[int(val), :, :]
        print(im_temp[1:50,1:10])
        self.im_disp.update_im(im_temp)

    def frame_features(self, frame, dt):

        img_label = self.labels[frame, :, :].copy()
        features_temp = []
        for j in range(self.channel_num):
            features_temp.append(regionprops(img_label, self.all_channels[j][frame, :, :]))

        self.counter += len(features_temp[0])
        feature_mat = np.zeros((len(features_temp[0]), self.feature_num))

        for j in range(len(features_temp[0])-1, 0, -1):

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
            img_label[img_label == cell_temp.label] = self.counter

            self.counter -= 1

        self.counter += len(features_temp[0])
        self.features = np.vstack((self.features, feature_mat))
        self.labels[frame, :, :] = img_label

    def extract_features(self, instance):

        self.clear_ui()
        self.current_ui = 5

        self.feat_message = Label(text='[b][color=000000]Extracting Features[/b][/color]', markup=True,
                                 size_hint=(.2, .05), pos_hint={'x': .4, 'y': .65})
        self.m_layout.add_widget(self.feat_message)

        self.channel_num = len(self.all_channels)
        self.feature_num = 20 + 3*(self.channel_num-1)
        self.features = np.zeros([1, self.feature_num])
        self.counter = 0

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
            print(self.labels)
            Window.bind(on_resize=self.training_p.update_size)


    def classify_cells(self, instance):

        self.training_data = self.s_param['training_data'][:, :]
        self.training_data = np.delete(self.training_data, 0, 0)

        clf = RandomForestClassifier(n_estimators=100)

        mask = np.sum(self.training_data[:, 12:17], axis=0) > 0
        inds = np.where(mask)[0]
        clf = clf.fit(self.training_data[:, 5:10], self.training_data[:, 12+inds].astype(bool))
        probs = clf.predict_proba(self.fov['feats'][:, 5:10])

        i = 0
        for p in probs:
            if len(p[0]) == 1:
                p = np.hstack([np.asarray(p), np.zeros((len(p), 1))])
            else:
                p = np.asarray(p)

            self.fov['feats'][:, 12 + inds[i]] = p[:, 1]
            i += 1

        self.progression_state(8)

    def run_tracking(self, instance):

        ''' Create matrix tracks, Col0 = ID from feature matrix; Col1 = Score difference; Col2 = total Score;
        Col3 = mitosis; Col4 = Track_id; Col5 = frame.

        Tracking parameters are; 0) distance penalty (smaller means
        less penalty); 1) max distance searched in previous frame; 2 & 3) weighting for adding segments based
        on probability of segments, p2*(s+1) - p3*(s); 4) mitosis penalty reduce the likelihood of mitosis,
        negative values increase the likelihood; 5) weighting for gaps in tracking; 6) max gap'''

        tracking_mat = self.fov['feats'][:, :]
        tracking_mat = np.vstack((np.zeros((1,tracking_mat.shape[1])),tracking_mat))

        states = np.zeros(tracking_mat.shape[0])
        states = states.astype(int)

        tracks = np.zeros([1, 8])
        self.fov['feats'][:, 18] = 1

        d_mat = tracking_c_tools.distance_mat(tracking_mat, int(self.frames), self.track_param)
        d_mat = d_mat[d_mat[:, 2].argsort(), :]

        d_mat= np.vstack((d_mat, np.zeros((1, d_mat.shape[1]))))

        s_mat = tracking_c_tools.swaps_mat(d_mat, self.frames)

        max_score = 1000.
        cum_score = 0.
        count = 1

        while max_score > 5.:

            print(0, count, max_score)

            score_mat = tracking_c_tools.forward_pass(tracking_mat, d_mat, s_mat, states, self.track_param)
            max_score = max(score_mat[:, 3])

            if max_score > 5.:

                cum_score += max_score
                track_temp, s_mat, states = tracking_c_tools.track_back(score_mat, states, s_mat)
                track_temp[:, 4] = count

                tracks, track_temp = tracking_c_tools.swap_test(tracks, track_temp, d_mat, count)

                tracks = np.vstack((tracks, track_temp))
                count += 1

        tracks = np.delete(tracks, 0, 0)

        iters = 3

        # Optimise tracks by iterating through removing and adding again

        for h in range(iters):
            for i in range(1, int(max(tracks[:, 4]))):

                replace_mask = tracks[:, 4] == i
                track_store = tracks[replace_mask, :]

                tracks = tracks[np.logical_not(replace_mask), :]
                for j in range(track_store.shape[0]):  # Remove track

                    states[int(track_store[j, 0])] -= 1

                    if j > 0:
                        ind1 = track_store[j - 1, 0]
                        ind2 = track_store[j, 0]

                        m1 = np.logical_and(s_mat[:, 1] == ind1, s_mat[:, 3] == ind2)
                        m2 = np.logical_and(s_mat[:, 2] == ind1, s_mat[:, 3] == ind2)
                        m3 = np.logical_and(s_mat[:, 1] == ind1, s_mat[:, 4] == ind2)
                        m4 = np.logical_and(s_mat[:, 2] == ind1, s_mat[:, 4] == ind2)

                        if any(m1):
                            s_mat[m1, 5] = 0
                            s_mat[m1, 7] = 0
                        if any(m2):
                            s_mat[m2, 6] = 0
                            s_mat[m2, 7] = 0
                        if any(m3):
                            s_mat[m3, 5] = 0
                            s_mat[m3, 8] = 0
                        if any(m4):
                            s_mat[m4, 6] = 0
                            s_mat[m4, 8] = 0

                score_mat = tracking_c_tools.forward_pass(tracking_mat, d_mat, s_mat, states, self.track_param)
                max_score = max(score_mat[:, 3])
                flag = False

                if max_score > track_store[-1, 2]:

                    cum_score = cum_score + max_score - track_store[-1, 2]
                    track_replace, s_mat, states = tracking_c_tools.track_back(score_mat, states, s_mat)
                    track_replace[:, 4] = i

                    tracks, track_replace = tracking_c_tools.swap_test(tracks, track_replace, d_mat, i)
                    tracks = np.vstack((tracks, track_replace))

                else:
                    tracks = np.vstack((tracks, track_store))

                    for j in range(track_store.shape[0]):

                        states[int(track_store[j, 0])] += 1

                        if j > 0:

                            ind1 = track_store[j - 1, 0]
                            ind2 = track_store[j - 1, 0]

                            m1 = np.logical_and(s_mat[:, 1] == ind1, s_mat[:, 3] == ind2)
                            m2 = np.logical_and(s_mat[:, 2] == ind1, s_mat[:, 3] == ind2)
                            m3 = np.logical_and(s_mat[:, 1] == ind1, s_mat[:, 4] == ind2)
                            m4 = np.logical_and(s_mat[:, 2] == ind1, s_mat[:, 4] == ind2)

                            if any(m1):
                                s_mat[m1, 5] = 1
                                s_mat[m1, 7] = 1
                            if any(m2):
                                s_mat[m2, 6] = 1
                                s_mat[m2, 7] = 1
                            if any(m3):
                                s_mat[m3, 5] = 1
                                s_mat[m3, 8] = 1
                            if any(m4):
                                s_mat[m4, 6] = 1
                                s_mat[m4, 8] = 1

                print(h, i, cum_score)
        print(states[1:200].astype(int))
        tracks[:, 0] = tracks[:, 0]-1

        # Color labels

        r = int(252 * np.random.rand()) + 3
        ind = tracks[0, 4]

        for j in range(tracks.shape[0]):

            if tracks[j, 4] != ind:
                ind = tracks[j, 4]
                r = int(252 * np.random.rand()) + 3

            self.fov['feats'][tracks[j, 0], 18] = r

            if tracks[j, 3] > 0:
                self.fov['feats'][tracks[j, 0], 18] = 5

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

    def tracking_ui(self, instance):

        if instance.state == 'down':

            self.tracking_p = TrackingUI(size_hint=(1., 1.), pos_hint={'x': .01, 'y': .01})
            self.add_widget(self.tracking_p)

            self.tracking_p.initialize(self.segment_channel, self.labels, self.frames)
            Window.bind(on_resize=self.tracking_p.update_size)

        else:

            self.tracking_p.remove()
            self.remove_widget(self.tracking_p)

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
                if g == 'labels':

                    # Load labels

                    labels = self.fov.require_dataset("labels", (self.frames, self.dims[0], self.dims[1]), dtype='i')
                    self.labels = labels[...]

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

                self.features[1:, 17:19] = 1
                self.features = self.features[np.argsort(self.features[:, 0]), :]

                # Delete if features already exists otherwise store extracted features as number of segments may change

                for g in self.fov:
                    if g == 'features':
                        del self.fov['features']

                self.fov.create_dataset("features", data=self.features)
                self.progression_state(6)

    def initialize(self):

        self.current_ui = 0
        self.track_param = np.asarray([0.05, 50, 1, 5, 0, 1, 3])
        self.progression = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        self.m_layout = FloatLayout(size=(Window.width, Window.height))
        self.layout1 = GridLayout(rows=1, padding=2, size_hint=(.9, .1), pos_hint={'x': .05, 'y': .01})

        btn1 = ToggleButton(text=' Load \nData', group='ui_choice')
        btn1.bind(on_press=self.data_ui)
        self.layout1.add_widget(btn1)

        # Progress bar widget

        self.layout2 = GridLayout(rows=1, padding=2, size_hint=(.9, .1), pos_hint={'x': .05, 'y': .5})
        self.pb = ProgressBar(max=1000, size_hint=(8., 1.), pos_hint={'x': .1, 'y': .6}, value=0)

        self.segment_flag = False
        self.feature_flag = False
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
