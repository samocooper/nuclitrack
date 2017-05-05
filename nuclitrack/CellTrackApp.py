import numpy as np
from functools import partial

from .uisegmentation import SegmentationUI, ViewSegment, BatchSegment
from .uitracking import TrackingUI, RunTracking
from .uitraining import TrainingUI, ClassifyCells
from .uiloading import LoadingUI
from .uifeatures import FeatureExtract

from kivy.app import App
from kivy.core.window import Window
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.togglebutton import ToggleButton
from kivy.uix.widget import Widget
from kivy.clock import Clock

class UserInterface(Widget):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.current_frame = 0
        self.parallel = False

        self.progression = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        self.m_layout = FloatLayout(size=(Window.width, Window.height))
        self.layout1 = GridLayout(rows=1, padding=5, size=(Window.width, Window.height / 10))

        btn1 = ToggleButton(text='Load\nData', group='ui_choice', halign='center', valign='middle')
        btn1.bind(on_press=self.loading_ui)
        btn1.bind(size=btn1.setter('text_size'))
        self.layout1.add_widget(btn1)

        # Progress bar widget

        self.segment_flag = False
        self.segment_flag_parallel = False
        self.feature_flag = False
        self.tracking_flag = False
        self.finish_flag = False

        self.iterations = 2
        Clock.schedule_interval(self.do_work, 0)

        self.current_widget = LoadingUI()
        Window.bind(on_resize=self.current_widget.update_size)

        with self.canvas:
            self.add_widget(self.m_layout)
            self.add_widget(self.layout1)
            self.add_widget(self.current_widget)

    def loading_ui(self, instance):
        if instance.state == 'down':

            self.remove_widget(self.current_widget)
            self.current_widget = LoadingUI()
            self.add_widget(self.current_widget)
            Window.bind(on_resize=self.current_widget.update_size)

    def segment_ui(self, instance):
        if instance.state == 'down':

            self.remove_widget(self.current_widget)
            self.params.require_dataset('seg_param', (11,), dtype='f')
            self.current_widget = SegmentationUI(images=self.images, frames=self.frames, channels=self.channels,
                                                 params=self.params['seg_param'][...])

            self.add_widget(self.current_widget)
            self.progression_state(3)
            Window.bind(on_resize=self.current_widget.update_size)

    def segment_movie(self, instance):
        if instance.state == 'down':

            self.remove_widget(self.current_widget)
            self.current_widget = BatchSegment(images=self.images[int(self.params['seg_param'][10])], params=self.params['seg_param'][...],
                                               frames=self.frames, parallel=self.parallel)
            self.add_widget(self.current_widget)
            self.labels = self.fov.require_dataset("labels", (self.frames, self.dims[0], self.dims[1]), dtype='i')

            if self.parallel == True:
                self.segment_flag_parallel = True

            else:
                self.count_scheduled = 0
                self.count_completed = 0
                self.segment_flag = True

    def view_segments(self, instance):
        if instance.state == 'down':

            self.remove_widget(self.current_widget)
            self.current_widget = ViewSegment(labels=self.labels, frames=self.frames)
            self.add_widget(self.current_widget)

    def segment_parallel(self, dt):

        self.current_widget.segment_parallel()
        self.labels[...] = self.current_widget.get()

        self.progression_state(4)
        self.progression_state(5)

    def finish_segmentation(self, dt):

        self.labels[...] = self.current_widget.get()

        self.progression_state(4)
        self.progression_state(5)

    def extract_features(self, instance):
        if instance.state == 'down':

            self.remove_widget(self.current_widget)
            self.current_widget = FeatureExtract(images=self.images, labels=self.labels[...], frames=self.frames,
                                                 channels=self.channels, dims=self.dims)
            self.add_widget(self.current_widget)
            self.feature_flag = True
            self.count_scheduled = 0
            self.count_completed = 0

    def save_features(self, dt):

        [features, self.labels[...]] = self.current_widget.get()

        # Delete if features already exists otherwise store extracted features as number of segments may change

        for g in self.fov:
            if g == 'features':
                del self.fov['features']

        self.fov.create_dataset("features", data=features)
        self.progression_state(6)

    def training_ui(self, instance):

        flag = False
        for g in self.fov:
            if g == 'features':
                flag = True

        if instance.state == 'down' and flag:

            self.remove_widget(self.current_widget)
            self.current_widget = TrainingUI(images=self.images[int(self.params['seg_param'][10])], labels=self.labels,
                                             features=self.fov['features'][...], frames=self.frames)
            self.add_widget(self.current_widget)
            Window.bind(on_resize=self.current_widget.update_size)

    def classify_cells(self, instance):
        if instance.state == 'down':

            self.remove_widget(self.current_widget)
            self.training_data = self.params['training_data'][:, :]
            self.current_widget = ClassifyCells(features=self.fov['features'][...], training_data=self.training_data)
            self.add_widget(self.current_widget)
            self.fov['features'][...] = self.current_widget.get()

            self.progression_state(8)

    def run_tracking(self, instance):

        self.remove_widget(self.current_widget)
        self.current_widget = RunTracking(features=self.fov['features'][...],
                                          track_param=self.params['track_param'][...], frames=self.frames)

        self.add_widget(self.current_widget)
        self.tracking_flag = True

    def add_tracks(self, dt):

        self.finish_flag = self.current_widget.add_track()

    def update_count(self, dt):

        self.current_widget.update_count()
        self.tracking_flag = True

        if self.finish_flag:

            self.tracking_flag = False
            self.tracks, self.fov['features'][...] = self.current_widget.get()

            # Delete if tracks already exists otherwise store extracted features

            for g in self.fov:
                if g == 'tracks':
                    del self.fov['tracks']

            for g in self.fov:
                if g == 'tracks_stored':
                    del self.fov['tracks_stored']

            self.fov.create_dataset("tracks", data=self.tracks)

            tracks_stored = np.zeros(int(max(self.tracks[:, 4])))
            self.fov.create_dataset("tracks_stored", data=tracks_stored)
            self.progression_state(9)

    def tracking_ui(self, instance):
        if instance.state == 'down':
            print(self.fov['features'][...].shape[1])
            self.remove_widget(self.current_widget)
            self.current_widget = TrackingUI(images=self.images, labels=self.labels, tracks=self.fov['tracks'][...],
                                             stored_tracks=self.fov['tracks_stored'][...],
                                             features=self.fov['features'][...], frames=self.frames, dims=self.dims,
                                             channels=self.channels)
            self.add_widget(self.current_widget)
            Window.bind(on_resize=self.current_widget.update_size)

    def progression_state(self, state):

        if state == 2 and self.progression[2] == 0 and self.progression[0] == 1 and self.progression[1] == 1:

            btn2 = ToggleButton(text='Segment',  group='ui_choice', halign='center', valign='middle')
            btn2.bind(on_press=self.segment_ui)
            btn2.bind(size=btn2.setter('text_size'))

            self.layout1.add_widget(btn2)
            self.progression[2] = 1

            for g in self.params:
                if g == 'seg_param':
                    state = 3

        if state == 3 and self.progression[3] == 0:

            btn3 = ToggleButton(text='Segment\nMovie',  group='ui_choice', halign='center', valign='middle')
            btn3.bind(on_press=self.segment_movie)
            btn3.bind(size=btn3.setter('text_size'))
            self.layout1.add_widget(btn3)

            self.progression[3] = 1

            for g in self.fov:
                if g == 'labels':

                    # Load labels

                    self.labels = self.fov.require_dataset("labels", (self.frames, self.dims[0], self.dims[1]), dtype='i')

                    state = 4

        if state == 4 and self.progression[4] == 0:

            btn4 = ToggleButton(text='View\nSegment',  group='ui_choice', halign='center', valign='middle')
            btn4.bind(on_press=self.view_segments)
            btn4.bind(size=btn4.setter('text_size'))

            self.layout1.add_widget(btn4)

            self.progression[4] = 1

            btn5 = ToggleButton(text='Extract\nFeatures', group='ui_choice', halign='center', valign='middle')
            btn5.bind(on_press=self.extract_features)
            btn5.bind(size=btn5.setter('text_size'))

            self.layout1.add_widget(btn5)

            self.progression[5] = 1

            for g in self.fov:
                if g == 'features':
                    state = 6


        if state == 6 and self.progression[6] == 0:

            btn6 = ToggleButton(text='Training\nData', group='ui_choice', halign='center', valign='middle')
            btn6.bind(on_press=self.training_ui)
            btn6.bind(size=btn6.setter('text_size'))

            self.layout1.add_widget(btn6)

            self.progression[6] = 1

            param_flag = True

            for g in self.params:
                if g == 'track_param':
                    param_flag = False

            if param_flag:
                self.params.create_dataset('track_param', data=np.asarray([0.05, 50, 1, 5, 0, 1, 3]))

            for g in self.params:
                if g == 'training_data':
                    state = 7

        if state == 7 and self.progression[7] == 0:

            btn7 = ToggleButton(text='Classify\nCells', group='ui_choice', halign='center', valign='middle')
            btn7.bind(on_press=self.classify_cells)
            btn7.bind(size=btn7.setter('text_size'))

            self.layout1.add_widget(btn7)

            self.progression[7] = 1

            for g in self.fov:
                if g == 'features':

                    cl = self.fov['features'][:, 12:15]

                    if sum(cl.flatten()) > 0:

                        state = 8

        if state == 8 and self.progression[8] == 0:
            btn8 = ToggleButton(text='Track\nCells', group='ui_choice', halign='center', valign='middle')
            btn8.bind(on_press=self.run_tracking)
            btn8.bind(size=btn8.setter('text_size'))

            self.layout1.add_widget(btn8)

            self.progression[8] = 1

            for g in self.fov:
                if g == 'tracks':
                    state = 9

        if state == 9 and self.progression[9] == 0:
            btn9 = ToggleButton(text='View\nTracks', group='ui_choice', halign='center', valign='middle')
            btn9.bind(on_press=self.tracking_ui)
            btn9.bind(size=btn9.setter('text_size'))

            self.layout1.add_widget(btn9)

            self.progression[9] = 1

    # Schedule heavy duty operations alongside loading bar updates

    def do_work(self, dt):

        self.canvas.ask_update()

        if self.segment_flag_parallel:
            Clock.schedule_once(self.segment_parallel, 0)
            self.segment_flag_parallel = False

        if self.segment_flag:

            Clock.schedule_once(self.current_widget.update_bar, 0)
            Clock.schedule_once(partial(self.current_widget.segment_im, self.count_scheduled), 0)
            self.count_scheduled += 1

            if self.count_scheduled == self.frames:
                self.segment_flag = False
                Clock.schedule_once(self.finish_segmentation)

        if self.feature_flag:

            Clock.schedule_once(self.current_widget.update_bar, 0)
            Clock.schedule_once(partial(self.current_widget.frame_features, self.count_scheduled), 0)
            self.count_scheduled += 1

            if self.count_scheduled == self.frames:

                self.feature_flag = False
                Clock.schedule_once(self.save_features, 0)

        if self.tracking_flag:

            if self.tracking_flag:
                self.tracking_flag = False
                Clock.schedule_once(self.add_tracks, 0)
                Clock.schedule_once(self.update_count, 0)

    def update_size(self, window, width, height):

        self.m_layout.width = width
        self.m_layout.height = height


class CellTrackApp(App):
    def build(self):

        ui = UserInterface()
        Window.clearcolor = (.8, .8, .8, 1)
        Window.bind(on_resize=ui.update_size)

        return ui
