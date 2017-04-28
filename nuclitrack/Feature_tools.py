import numpy as np
from skimage.measure import regionprops

from kivy.core.window import Window
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.uix.widget import Widget
from kivy.uix.progressbar import ProgressBar


class FeatureExtract(Widget):

    def __init__(self, images=None, labels=None, frames=None, channels=None, dims=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.images = images
        self.labels = labels
        self.frames = frames
        self.channels = channels
        self.dims = dims

        self.layout = FloatLayout(size=(Window.width, Window.height))
        self.feat_message = Label(text='[b][color=000000]Extracting Features[/b][/color]', markup=True,
                                  size_hint=(.2, .05), pos_hint={'x': .4, 'y': .65})

        self.feature_num = 20 + 3 * (self.channels - 1)
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

        img_label = self.labels[frame, :, :].copy()
        features_temp = []

        for j in range(self.channels):
            features_temp.append(regionprops(img_label, self.images[j][frame, :, :]))

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

            for k in range(1, self.channels):
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

    def get(self):

        self.feat_message.text = '[b][color=000000]Features Extracted[/b][/color]'

        self.features[1:, 17:19] = 1
        self.features = self.features[np.argsort(self.features[:, 0]), :]

        return [self.features, self.labels]
