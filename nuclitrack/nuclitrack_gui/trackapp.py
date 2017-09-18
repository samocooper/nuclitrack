from functools import partial
import numpy as np

from kivy.app import App
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.uix.gridlayout import GridLayout
from kivy.uix.widget import Widget

from .uitracking import TrackingUI, RunTracking
from .uifeatures import FeatureExtract
from .uiloading import LoadingUI
from .uisegmentation import SegmentationUI, ViewSegment, BatchSegment
from .uitraining import TrainingUI, ClassifyCells
from ..nuclitrack_guitools import guitools

class UserInterface(Widget):

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

        self.current_frame = 0
        self.parallel = False
        self.ring_flag = False

        # Progression state defines how many steps the user has gone through
        # Automatically updates on loading of HDF5 file

        self.progression = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        # Layout for binding progression buttons

        self.master_btns = GridLayout(rows=1, padding=5, size=(Window.width, Window.height / 10))

        # Add first button for loading data

        guitools.add_tbtn(layout=self.master_btns, text='Load Data', group='uis', func=self.loading_ui)

        # Flags for performing work using scheduler. System works by scheduling a small section of work then updating
        # progress bar. A flag is set to true, work is performed and the flag set to false. On the next kivy frame the
        # loading bar  updates and this sets the flag to True. This appears to provide a very effective lock on
        # preventing more work being scheduled, and blocking loading bar update.

        self.flags = {'segment': False, 'segment_parallel': False, 'feat': False,
                      'track': False, 'finish': False, 'cancel': False}

        # On each kivy frame test if work needs to be performed.

        Clock.schedule_interval(self.do_work, 0)

        # Attach the first UI for loading data. Current widget is used for all of the main nuclitrack_gui,
        # This allows easy clearing of the previous UI

        self.current_widget = LoadingUI()
        Window.bind(on_resize=self.current_widget.update_size)

        with self.canvas:
            self.add_widget(self.master_btns)
            self.add_widget(self.current_widget)

    # Change the current UI

    def change_widget(self, new_widget):

        self.remove_widget(self.current_widget)
        self.current_widget = new_widget
        self.add_widget(self.current_widget)
        Window.bind(on_resize=self.current_widget.update_size)

    # UI for loading data

    def loading_ui(self, instance):
        if instance.state == 'down':
            self.change_widget(LoadingUI())

    # UI for choosing segmentation parameters, these are stored in params['seg_param'] HDF5 file

    def segment_ui(self, instance):
        if instance.state == 'down':

            self.params.require_dataset('seg_param', (18,), dtype='f')

            if not any(self.params['seg_param'][15:18]):
                self.params['seg_param'][15] = 1

            self.change_widget(SegmentationUI(movie=self.movie, params=self.params))
            self.progression_state(3)

    # Widget for segmenting images, includes loading bar and schedules segmentation fucntion

    def segment_movie(self, instance):
        if instance.state == 'down':

            self.remove_widget(self.current_widget)

            if 'labels' in self.fov:
                del self.fov['labels']
            self.labels = self.fov.create_dataset("labels", self.movie.shape)

            self.change_widget(BatchSegment(movie=self.movie, params=self.params,
                                            labels=self.labels, parallel=self.parallel))

            # Set segmentation flags to True to start performing work

            if self.parallel == True:
                self.flags['segment_parallel'] = True

            else:
                self.count_scheduled = 0
                self.count_completed = 0
                self.flags['segment'] = True

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

            self.change_widget(ViewSegment(movie=self.movie, labels=self.labels))

    # Widget that brings up loading bar for feature extraction

    def extract_features(self, instance):
        if instance.state == 'down':

            self.change_widget(FeatureExtract(movie=self.movie, labels=self.labels[...], ring_flag=self.ring_flag))

            self.flags['feat'] = True
            self.count_scheduled = 0
            self.count_completed = 0

    # Collect results of feature extraction and save them to HDF5 file overwriting any previous results.

    def save_features(self, dt):

        [features, self.labels[...]] = self.current_widget.get()

        # Delete if features already exists otherwise store extracted features as number of segments may change.

        if 'features' in self.fov:
            del self.fov['features']
        self.features = self.fov.create_group('features')
        self.features.create_dataset("tracking", data=features['tracking'])
        self.features.create_dataset("data", data=features['data'])

        self.progression_state(6)

    # UI for selecting training classes from segmented cells

    def training_ui(self, instance):

        if instance.state == 'down' and 'features' in self.fov:

            store = 'training' in self.params
            self.change_widget(TrainingUI(movie=self.movie, labels=self.labels, features=self.fov['features'],
                                     params=self.params['track_param'][...], stored=store))

    # UI for classifying cells based upon training data

    def classify_cells(self, instance):
        if instance.state == 'down':

            self.change_widget(ClassifyCells(features=self.fov['features'], training=self.params['training']))
            self.features = self.current_widget.get()

            self.progression_state(8)

    # Widget that performs tracing and loads up display to show number of tracked cells.

    def run_tracking(self, instance):

        self.change_widget(RunTracking(features=self.fov['features']['tracking'][...],
                                          track_param=self.params['track_param'][...], frames=self.movie.frames))
        self.flags['track'] = True

    def add_tracks(self, dt):
        if not self.flags['cancel']:
            self.flags['finish'] = self.current_widget.add_track()

    # Functions that schedules updates to the tracking Widget display and at the end collects results and saves them

    def update_count(self, dt):
        if not self.flags['cancel']:

            self.current_widget.update_count()
            self.flags['track'] = True

            if self.flags['finish']:

                self.flags['track'] = False
                cancel = self.current_widget.test_cancel()

                # Collect results and save to DHF5 file if tracking has not been cancelled

                if not cancel:
                    self.tracks, self.fov['features']['tracking'][...] = self.current_widget.get()

                    # Delete if tracks already exists otherwise store extracted features

                    if 'tracks' in self.fov:
                        del self.fov['tracks']
                    self.fov.create_dataset("tracks", data=self.tracks)

                    if 'tracks_stored' in self.fov:
                        del self.fov['tracks_stored']
                    self.fov.create_dataset("tracks_stored", data=np.zeros(int(max(self.tracks[:, 4]))))

                    self.progression_state(9)

    # UI for inspecting and ammending tracks

    def tracking_ui(self, instance):
        if instance.state == 'down':
            self.change_widget(TrackingUI(movie=self.movie, labels=self.labels, tracks=self.fov['tracks'],
                                             stored_tracks=self.fov['tracks_stored'],
                                             features=self.fov['features']))

    # Function that determines how far the user has proceeded. When this function is called with the next number the
    # next button is added to the progression button layout. On loading of the HDF5 data and parameter files, this
    # function determines how far the user has progressed and returns them to that state.

    def progression_state(self, state):

        if state == 2 and self.progression[2] == 0 and self.progression[0] == 1 and self.progression[1] == 1:

            guitools.add_tbtn(layout=self.master_btns, text='Segment', group='uis', func=self.segment_ui)
            
            self.progression[2] = 1

            if 'seg_param' in self.params:
                state = 3

        if state == 3 and self.progression[3] == 0:

            guitools.add_tbtn(layout=self.master_btns, text='Segment Movie', group='uis', func=self.segment_movie)

            self.progression[3] = 1

            if 'labels' in self.fov:

                # Load labels
                self.labels = self.fov['labels']
                state = 4

        if state == 4 and self.progression[4] == 0:

            guitools.add_tbtn(layout=self.master_btns, text='View Segment', group='uis', func=self.view_segments)
            
            self.progression[4] = 1

            guitools.add_tbtn(layout=self.master_btns, text='Extract Features', group='uis', func=self.extract_features)
            
            self.progression[5] = 1

            if 'features' in self.fov:
                self.features = self.fov['features']
                state = 6

        if state == 6 and self.progression[6] == 0:

            guitools.add_tbtn(layout=self.master_btns, text='Training Data', group='uis', func=self.training_ui)
            self.progression[6] = 1

            param_flag = True

            if 'track_param' in self.params:
                param_flag = False

            if param_flag:
                self.params.create_dataset('track_param', data=np.asarray([0.05, 50, 1, 5, 0, 1, 3]))

            if 'training' in self.params:
                if self.params['training']['data'].shape[0] > 1:
                    state = 7

        if state == 7 and self.progression[7] == 0:

            guitools.add_tbtn(layout=self.master_btns, text='Classify Cells', group='uis', func=self.classify_cells)

            self.progression[7] = 1

            cl = self.features['tracking'][:, 11:14]

            if sum(cl.flatten()) > 0:
                state = 8

        if state == 8 and self.progression[8] == 0:

            guitools.add_tbtn(layout=self.master_btns, text='Track Cells', group='uis', func=self.run_tracking)

            self.progression[8] = 1

            if 'tracks' in self.fov:
                state = 9

        if state == 9 and self.progression[9] == 0:

            guitools.add_tbtn(layout=self.master_btns, text='View Tracks', group='uis', func=self.tracking_ui)

            self.progression[9] = 1

    # Schedule heavy duty operations alongside loading bar updates

    def do_work(self, dt):

        try:
            self.canvas.ask_update()

            if self.flags['segment_parallel']:
                Clock.schedule_once(self.segment_parallel, 0)
                self.flags['segment_parallel'] = False

            if self.flags['segment']:

                Clock.schedule_once(self.current_widget.update_bar, 0)
                Clock.schedule_once(partial(self.current_widget.segment_im, self.count_scheduled), 0)
                self.count_scheduled += 1

                if self.count_scheduled == self.movie.frames:
                    self.flags['segment'] = False
                    Clock.schedule_once(self.finish_segmentation)

            if self.flags['feat']:

                Clock.schedule_once(self.current_widget.update_bar, 0)
                Clock.schedule_once(partial(self.current_widget.frame_features, self.count_scheduled), 0)
                self.count_scheduled += 1

                if self.count_scheduled == self.movie.frames:

                    self.flags['feat'] = False
                    Clock.schedule_once(self.save_features, 0)

            if self.flags['track']:

                self.flags['track'] = False

                Clock.schedule_once(self.add_tracks, 0)
                Clock.schedule_once(self.update_count, 0)

        except AttributeError:
            guitools.notify_msg('Please allow process to finish')

    def update_size(self, window, width, height):

        self.master_btns.width = width
        self.master_btns.height = height / 10


class CellTrackApp(App):
    def build(self):

        ui = UserInterface()
        Window.clearcolor = (.8, .8, .8, 1)
        Window.bind(on_resize=ui.update_size)

        return ui