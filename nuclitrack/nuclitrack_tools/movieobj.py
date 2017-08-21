from PIL import Image
import numpy as np


class MovieObj(object):

    loaded = False

    def __init__(self, file_list):

        try:
            channels = len(file_list)
            frames = len(file_list[0])

            im_temp = Image.open(file_list[0][0])
            im_test = np.asarray(im_temp, dtype='float')

            dims = im_test.shape

            min_vals = []
            max_vals = []

            for j in range(len(file_list)):

                im_temp = Image.open(file_list[j][0])
                im_test = np.asarray(im_temp, dtype='float')

                max_val = np.max(im_test)
                min_val = np.min(im_test)

                for i in range(frames):

                    im_temp = Image.open(file_list[j][i])
                    im = np.asarray(im_temp, dtype='float')

                    im = im.astype(float)

                    if np.max(im) > max_val:
                        max_val = np.max(im)
                    if np.min(im) < min_val:
                        min_val = np.min(im)

                min_vals.append(min_val)
                max_vals.append(max_val)

            self.file_list = file_list
            self.dims = dims
            self.channels = channels
            self.frames = frames
            self.min_vals = min_vals
            self.max_vals = max_vals
            self.loaded = True
            self.shape = (frames, dims[0], dims[1])

        except FileNotFoundError:

            print('File not found: ' + file_list[0][0] + '\nDirectory may have changed')

    def read_im(self, channel, frame):

        pil_im = Image.open(self.file_list[channel][frame])
        temp_im = np.asarray(pil_im, dtype='float').copy()
        temp_im -= self.min_vals[channel]
        temp_im /= self.max_vals[channel]
        im = temp_im

        return im

    def comb_im(self, channels, frame):

        im = np.zeros(self.dims)

        for i in range(len(channels)):
            if channels[i]:
                pil_im = Image.open(self.file_list[i][frame])
                temp_im = np.asarray(pil_im, dtype='float').copy()
                temp_im -= self.min_vals[i]
                temp_im /= self.max_vals[i]
                im += temp_im

        return im

    def read_raw(self, channel, frame):

        pil_im = Image.open(self.file_list[channel][frame])
        im = np.asarray(pil_im, dtype='float').copy()

        return im