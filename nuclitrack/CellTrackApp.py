import numpy as np
from functools import partial

from .uisegmentation import SegmentationUI, ViewSegment, BatchSegment
from .uitracking import TrackingUI, RunTracking
from .uitraining import TrainingUI, ClassifyCells
from .uiloading import LoadingUI
from .uifeatures import FeatureExtract

from kivy.app import App
from kivy.core.window import Window
from kivy.uix.gridlayout import GridLayout
from kivy.uix.togglebutton import ToggleButton
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.uix.widget import Widget
from kivy.clock import Clock


class UserInterface(Widget):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.current_frame = 0
        self.parallel = False
        self.ring_flag= False
        # Set of values that are used by file loading function to store data on image series,
        # Prevents need to load images into RAM

        self.dims = []
        self.min_vals = []
        self.max_vals = []
        self.file_list = []

        # Progression state defines how many steps the user has gone through
        # Automatically updates on loading of HDF5 file

        self.progression = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        # Layout for binding progression buttons

        self.layout1 = GridLayout(rows=1, padding=5, size=(Window.width, Window.height / 10))

        # Add first button for loading data

        self.btn1 = ToggleButton(text='Load\nData', group='ui_choice', halign='center', valign='middle')
        self.btn1.bind(on_press=self.loading_ui)
        self.btn1.bind(size=self.btn1.setter('text_size'))
        self.layout1.add_widget(self.btn1)

        # Flags for performing work using scheduler. System works by scheduling a small section of work then updating
        # progress bar. A flag is set to true, work is performed and the flag set to false. On the next kivy frame the
        # loading bar  updates and this sets the flag to True. This appears to provide a very effective lock on
        # preventing more work being scheduled, and blocking loading bar update.

        self.segment_flag = False
        self.segment_flag_parallel = False
        self.feature_flag = False
        self.tracking_flag = False
        self.finish_flag = False

        # On each kivy frame test if work needs to be performed.

        Clock.schedule_interval(self.do_work, 0)

        # Attach the first UI for loading data. Current widget is used for all of the main UIs,
        # This allows easy clearing of the previous UI

        self.current_widget = LoadingUI()
        Window.bind(on_resize=self.current_widget.update_size)

        with self.canvas:
            self.add_widget(self.layout1)
            self.add_widget(self.current_widget)

    # Load image data from files, modified such that only file list is loaded to reduce RAM load.

    def error_message(self, message):

        error_msg = Popup(title='Error message',
                          content=Label(text=message),
                          size_hint=(0.6, 0.3))
        error_msg.open()

    def loading_ui(self, instance):
        if instance.state == 'down':

            self.remove_widget(self.current_widget)
            self.current_widget = LoadingUI()
            self.add_widget(self.current_widget)
            Window.bind(on_resize=self.current_widget.update_size)

    # UI for choosing segmentation parameters, these are stored in params['seg_param'] HDF5 file

    def segment_ui(self, instance):
        if instance.state == 'down':

            self.params.require_dataset('seg_param', (11,), dtype='f')

            if self.params['seg_param'][10] >= self.channels:
                self.params['seg_param'][10] = 0

            self.remove_widget(self.current_widget)
            self.current_widget = SegmentationUI(file_list=self.file_list, min_vals=self.min_vals,
                                                 max_vals=self.max_vals, frames=self.frames,
                                                 channels=self.channels, params=self.params['seg_param'][...])

            self.add_widget(self.current_widget)
            self.progression_state(3)
            Window.bind(on_resize=self.current_widget.update_size)

    # Widget for segmenting images, includes loading bar and schedules segmentation fucntion

    def segment_movie(self, instance):
        if instance.state == 'down':

            self.remove_widget(self.current_widget)
            channel = int(self.params['seg_param'][10])

            for g in self.fov:
                if g == 'labels':
                    del self.fov['labels']
            self.labels = self.fov.create_dataset("labels", (self.frames, self.dims[0], self.dims[1]))

            self.current_widget = BatchSegment(file_list=self.file_list[channel],
                                               min_val=self.min_vals[channel],
                                               max_val=self.max_vals[channel],
                                               params=self.params['seg_param'][...], frames=self.frames,
                                               labels=self.labels,
                                               parallel=self.parallel)
            self.add_widget(self.current_widget)

            # Set segmentation flags to True to start performing work

            if self.parallel == True:
                self.segment_flag_parallel = True

            else:
                self.count_scheduled = 0
                self.count_completed = 0
                self.segment_flag = True

    # Function to schedule parallel segmentation and collect results from segmentation widget

    def segment_parallel(self, dt):

        self.current_widget.segment_parallel()
        self.labels[...] = self.current_widget.get()

        self.progression_state(4)
        self.progression_state(5)

    # Function that gets labels from segmentation widget

    def finish_segmentation(self, dt):

        self.labels[...] = self.current_widget.get()

        self.progression_state(4)
        self.progression_state(5)

    # UI for visualising segmentation results and exporting labels to series of .tif images

    def view_segments(self, instance):
        if instance.state == 'down':

            self.remove_widget(self.current_widget)
            self.current_widget = ViewSegment(labels=self.labels, frames=self.frames)
            self.add_widget(self.current_widget)
            Window.bind(on_resize=self.current_widget.update_size)

    # Widget that bings up loading bar for feature extraction

    def extract_features(self, instance):
        if instance.state == 'down':

            self.remove_widget(self.current_widget)
            self.current_widget = FeatureExtract(file_list=self.file_list, labels=self.labels[...], frames=self.frames,
                                                 channels=self.channels, dims=self.dims, ring_flag=self.ring_flag)
            self.add_widget(self.current_widget)
            self.feature_flag = True
            self.count_scheduled = 0
            self.count_completed = 0

    # Collect results of feature extraction and save them to HDF5 file overwriting any previous results.

    def save_features(self, dt):

        [features, self.labels[...]] = self.current_widget.get()

        # Delete if features already exists otherwise store extracted features as number of segments may change.

        for g in self.fov:
            if g == 'features':
                del self.fov['features']

        self.features = self.fov.create_group('features')
        self.features.create_dataset("tracking", data=features['tracking'])
        self.features.create_dataset("data", data=features['data'])

        self.progression_state(6)

    # UI for selecting training classes from segmented cells

    def training_ui(self, instance):

        flag = False
        for g in self.fov:
            if g == 'features':
                flag = True
        if instance.state == 'down' and flag:
            store = False
            for g in self.params:
                if g == 'training':
                    store = True

            self.remove_widget(self.current_widget)
            self.current_widget = TrainingUI(file_list=self.file_list[int(self.params['seg_param'][10])],
                                             labels=self.labels, features=self.fov['features'], frames=self.frames,
                                             params=self.params['track_param'][...], stored=store)
            self.add_widget(self.current_widget)
            Window.bind(on_resize=self.current_widget.update_size)

    # UI for classifying cells based upon training data

    def classify_cells(self, instance):
        if instance.state == 'down':

            self.remove_widget(self.current_widget)
            self.current_widget = ClassifyCells(features=self.fov['features'], training=self.params['training'])
            self.add_widget(self.current_widget)
            self.features = self.current_widget.get()

            self.progression_state(8)

    # Widget that performs tracing and loads up display to show number of tracked cells.

    def run_tracking(self, instance):

        self.remove_widget(self.current_widget)
        self.current_widget = RunTracking(features=self.fov['features']['tracking'][...],
                                          track_param=self.params['track_param'][...], frames=self.frames)

        self.add_widget(self.current_widget)
        self.tracking_flag = True
        self.cancel_flag = False

    def add_tracks(self, dt):
        if not self.cancel_flag:
            self.finish_flag = self.current_widget.add_track()

    # Functions that schedules updates to the tracking Widget display and at the end collects results and saves them

    def update_count(self, dt):
        if not self.cancel_flag:

            self.current_widget.update_count()
            self.tracking_flag = True

            if self.finish_flag:

                self.tracking_flag = False
                cancel = self.current_widget.test_cancel()

                if not cancel:
                    self.tracks, self.fov['features']['tracking'][...] = self.current_widget.get()

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

    # UI for inspecting and ammending tracks

    def tracking_ui(self, instance):
        if instance.state == 'down':
            self.remove_widget(self.current_widget)
            self.current_widget = TrackingUI(file_list=self.file_list, labels=self.labels, tracks=self.fov['tracks'],
                                             stored_tracks=self.fov['tracks_stored'],
                                             features=self.fov['features'], frames=self.frames, dims=self.dims,
                                             channels=self.channels)
            self.add_widget(self.current_widget)
            Window.bind(on_resize=self.current_widget.update_size)

    # Function that determines how far the user has proceeded. When this function is called with the next number the
    # next button is added to the progression button layout. On loading of the HDF5 data and parameter files, this
    # function determines how far the user has progressed and returns them to that state.

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

                    self.labels = self.fov['labels']
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
                    self.features = self.fov['features']
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
                if g == 'training':
                    if self.params['training']['data'].shape[0] > 1:
                        state = 7

        if state == 7 and self.progression[7] == 0:

            btn7 = ToggleButton(text='Classify\nCells', group='ui_choice', halign='center', valign='middle')
            btn7.bind(on_press=self.classify_cells)
            btn7.bind(size=btn7.setter('text_size'))

            self.layout1.add_widget(btn7)
            self.progression[7] = 1
            cl = self.features['tracking'][:, 11:14]

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

        try:
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
        except AttributeError:

            self.error_message('Please allow process to finish')

    def update_size(self, window, width, height):

        self.layout1.width = width
        self.layout1.height = height / 10


class CellTrackApp(App):
    def build(self):

        ui = UserInterface()
        Window.clearcolor = (.8, .8, .8, 1)
        Window.bind(on_resize=ui.update_size)

        return ui