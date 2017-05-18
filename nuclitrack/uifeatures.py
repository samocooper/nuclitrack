import numpy as np
from kivy.core.window import Window
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.uix.widget import Widget
from kivy.uix.progressbar import ProgressBar
from . import extractfeatures

class FeatureExtract(Widget):

    def __init__(self, file_list, labels, frames, channels, dims, ring_flag=False, **kwargs):
        super().__init__(**kwargs)

        self.ring_flag = ring_flag
        self.file_list = file_list
        self.labels = labels
        self.frames = frames
        self.channels = channels
        self.dims = dims

        self.layout = FloatLayout(size=(Window.width, Window.height))
        self.feat_message = Label(text='[b][color=000000]Extracting Features[/b][/color]', markup=True,
                                  size_hint=(.2, .05), pos_hint={'x': .4, 'y': .65})

        self.features = dict()
        self.features['tracking'] = np.zeros([1, 13])
        self.features['data'] = np.zeros([1, 22])

        self.counter = 1

        self.layout2 = GridLayout(rows=1, padding=2, size_hint=(.9, .1), pos_hint={'x': .05, 'y': .5})
        self.pb = ProgressBar(max=1000, size_hint=(8., 1.), pos_hint={'x': .1, 'y': .6}, value= 1000 / self.frames)
        self.layout2.add_widget(self.pb)

        with self.canvas:

            self.add_widget(self.layout)
            self.layout.add_widget(self.feat_message)
            self.layout.add_widget(self.layout2)

    def update_bar(self, dt):
        self.pb.value += 1000 / self.frames

    def frame_features(self, frame, dt):

        files = []
        for i in range(self.channels):
            files.append(self.file_list[i][frame])

        features_temp, new_labels, self.counter = extractfeatures.framefeatures(files, self.labels[frame, :, :],
                                                                                self.counter, ring_flag=self.ring_flag)
        features_temp['tracking'][:, 1] = frame
        self.features['tracking'] = np.vstack((self.features['tracking'], features_temp['tracking']))
        self.features['data'] = np.vstack((self.features['data'], features_temp['data']))

        self.labels[frame, :, :] = new_labels

    def get(self):

        self.feat_message.text = '[b][color=000000]Features Extracted[/b][/color]'
        inds = np.argsort(self.features['tracking'][:, 0])
        self.features['tracking'] = self.features['tracking'][inds, :]
        self.features['data'] = self.features['data'][inds, :]
        self.features['tracking'][1:, 5] = 1.

        return [self.features, self.labels]
