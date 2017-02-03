import numpy as np

from kivy.graphics.texture import Texture
from kivy.graphics import Rectangle
from kivy.uix.widget import Widget

import cmaps
import image_plot

class ImDisplay(Widget):

    ''' Class for displaying numpy matrices as widgets on the canvas'''

    def create_texture(self, mat):

        dims = np.shape(mat)  # Dimensions of the Image to show
        self.texture = Texture.create(size=(dims[1], dims[0]), colorfmt='rgb')  # Create texture sized to image

    def display(self, mat, c_map):

        self.cmap = cmaps.color_map(c_map)

        m = mat.flatten()
        m = image_plot.scale_im_plot(m)

        im = image_plot.color_map(m, self.cmap)

        arr = np.asarray(im, dtype=np.uint8)
        self.texture.blit_buffer(arr.tostring(), colorfmt='rgb', bufferfmt='ubyte')

        with self.canvas:
            self.im = Rectangle(texture=self.texture, size=self.size, pos=self.pos)

        self.bind(pos=self.update_im, size=self.update_im)

    def update_im(self, *args):

        self.im.pos = self.pos
        self.im.size = self.size

    def update(self, mat):
        m = mat.flatten()

        m = image_plot.scale_im_plot(m)
        im = image_plot.color_map(m, self.cmap)

        arr = np.asarray(im, dtype=np.uint8)
        self.texture.blit_buffer(arr.tostring(), colorfmt='rgb', bufferfmt='ubyte')

    def indexed_display(self, mat, c_map, mapping):
        self.cmap = cmaps.color_map(c_map)

        m = mat.flatten()
        im = image_plot.indexed_color_map(m.astype(int), self.cmap, mapping)

        arr = np.asarray(im, dtype=np.uint8)
        self.texture.blit_buffer(arr.tostring(), colorfmt='rgb', bufferfmt='ubyte')

        with self.canvas:
            self.im = Rectangle(texture=self.texture, size=self.size, pos=self.pos)

        self.bind(pos=self.update_im, size=self.update_im)

    def indexed_update(self, mat, mapping):
        m = mat.flatten()
        im = image_plot.indexed_color_map(m.astype(int), self.cmap, mapping)

        arr = np.asarray(im, dtype=np.uint8)
        self.texture.blit_buffer(arr.tostring(), colorfmt='rgb', bufferfmt='ubyte')


    def remove(self):
        self.canvas.clear()
