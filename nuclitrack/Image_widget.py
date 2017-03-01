import numpy as np
from kivy.graphics import Rectangle
from kivy.graphics.texture import Texture
from kivy.uix.widget import Widget

from .cmaps import color_map
import numpy_to_image

class ImDisplay(Widget):

    ''' Class for displaying numpy matrices as widgets on the canvas. Scaling is performed on the image by default
    otherwise image values must lie between 0 and 255. Mapping to rgb color scheme is then performed in Cython compiled
    code.'''

    def create_im(self, mat, c_map,  scale=True):

        dims = np.shape(mat)  # Dimensions of the Image to show
        self.texture = Texture.create(size=(dims[1], dims[0]), colorfmt='rgb')  # Create texture sized to image

        self.scale = scale
        self.cmap = color_map(c_map)  # Specify the colour map to use, these are stored in the cmaps file

        m = mat.flatten()
        m = m.astype(float)  # Type as float, greater precision when scaling benchmarks faster than int


        if scale:
            im = numpy_to_image.scale_im(m, len(self.cmap) // 3 - 1)  # Scale image between 0 and 255
        else:
            im = m

        im = im.astype(int)
        im = numpy_to_image.mat_to_im(im, self.cmap)  # Map scaled image to colormap
        arr = np.asarray(im, dtype=np.uint8)
        self.texture.blit_buffer(arr.tostring(), colorfmt='rgb', bufferfmt='ubyte')

        with self.canvas:
            self.im = Rectangle(texture=self.texture, size=self.size, pos=self.pos)  # Add image to canvas

        self.bind(pos=self.update_size, size=self.update_size)  # Maintain image size on scaling of parent layout

    def update_im(self, mat):

        m = mat.flatten()
        m = m.astype(float)

        if self.scale:
            im = numpy_to_image.scale_im(m, len(self.cmap) // 3 - 1)
        else:
            im = m

        im = im.astype(int)
        im = numpy_to_image.mat_to_im(im, self.cmap)

        arr = np.asarray(im, dtype=np.uint8)
        self.texture.blit_buffer(arr.tostring(), colorfmt='rgb', bufferfmt='ubyte')

    def update_size(self, *args):

        self.im.pos = self.pos
        self.im.size = self.size

class IndexedDisplay(Widget):

    ''' Class for displaying numpy matrices as widgets on the canvas. Unlike ImDisplay, indexed display takes in an
    input colour mapping matrix, this maps matrix values to colour map indexes, useful when multiple values in numpy
    matrix must point to the same colour '''

    def create_im(self, mat, c_map, mapping):

        dims = np.shape(mat)  # Dimensions of the Image to show
        self.texture = Texture.create(size=(dims[1], dims[0]), colorfmt='rgb')  # Create texture sized to image


        self.cmap = color_map(c_map)

        m = mat.flatten()
        im = numpy_to_image.indexed_mat_to_im(m.astype(int), self.cmap, mapping)

        arr = np.asarray(im, dtype=np.uint8)
        self.texture.blit_buffer(arr.tostring(), colorfmt='rgb', bufferfmt='ubyte')

        with self.canvas:
            self.im = Rectangle(texture=self.texture, size=self.size, pos=self.pos)

        self.bind(pos=self.update_size, size=self.update_size)

    def update_im(self, mat, mapping):

        m = mat.flatten()
        im = numpy_to_image.indexed_mat_to_im(m.astype(int), self.cmap, mapping)

        arr = np.asarray(im, dtype=np.uint8)
        self.texture.blit_buffer(arr.tostring(), colorfmt='rgb', bufferfmt='ubyte')

    def update_size(self, *args):
        self.im.pos = self.pos
        self.im.size = self.size

