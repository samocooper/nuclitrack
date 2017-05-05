import numpy as np
from kivy.core.window import Window
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.uix.widget import Widget
from kivy.uix.progressbar import ProgressBar
from . import extractfeatures

class FeatureExtract(Widget):

    def __init__(self, images=None, labels=None, frames=None, channels=None, dims=None, **kwargs):
        super().__init__(**kwargs)

        self.images = images
        self.labels = labels
        self.frames = frames
        self.channels = channels
        self.dims = dims

        self.layout = FloatLayout(size=(Window.width, Window.height))
        self.feat_message = Label(text='[b][color=000000]Extracting Features[/b][/color]', markup=True,
                                  size_hint=(.2, .05), pos_hint={'x': .4, 'y': .65})

        self.feature_num = 21 + 3 * (self.channels - 1)
        self.features = np.zeros([1, self.feature_num])
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

        feature_images = [self.labels[frame, :, :]]
        for i in range(self.channels):
            feature_images.append(self.images[i][frame, :, :])

        feature_mat, new_labels, self.counter = extractfeatures.framefeatures(feature_images, self.feature_num, self.counter)
        feature_mat[:, 1] = frame
        self.features = np.vstack((self.features, feature_mat))
        self.labels[frame, :, :] = new_labels

    def get(self):

        self.feat_message.text = '[b][color=000000]Features Extracted[/b][/color]'

        self.features[1:, 17:19] = 1
        self.features = self.features[np.argsort(self.features[:, 0]), :]

        return [self.features, self.labels]
