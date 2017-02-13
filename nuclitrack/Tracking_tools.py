import numpy as np
import h5py

from kivy.graphics import Ellipse, Color
from kivy.uix.widget import Widget
from kivy.uix.slider import Slider
from kivy.uix.togglebutton import ToggleButton, Button
from kivy.uix.gridlayout import GridLayout
from kivy.core.window import Window, Keyboard
from kivy.uix.floatlayout import FloatLayout
from functools import partial
from kivy.uix.textinput import TextInput

from skimage.external import tifffile
from scipy.spatial import distance

from .Image_widget import ImDisplay, IndexedDisplay
from .Graph import Graph, SmoothLinePlot


class CellMark(Widget):
    def draw_dot(self, cell_center, dims, r, g, b, d):
        self.d = d
        cell_center[0] = cell_center[0] / dims[1]
        cell_center[1] = cell_center[1] / dims[0]

        self.cell_center = cell_center

        ds = self.height / self.d
        xpos = (cell_center[0] * self.width) + self.pos[0] - ds / 2
        ypos = (cell_center[1] * self.height) + self.pos[1] - ds / 2

        with self.canvas:
            Color(r, g, b)
            self.dot = Ellipse(size=(ds, ds), pos=(xpos, ypos))

        self.bind(pos=self.update_im, size=self.update_im)

    def update_im(self, *args):
        ds = self.height / self.d
        xpos = (self.cell_center[0] * self.width) + self.pos[0] - ds / 2
        ypos = (self.cell_center[1] * self.height) + self.pos[1] - ds / 2

        self.dot.pos = [xpos, ypos]
        self.dot.size = [ds, ds]

    def remove_dot(self):
        self.canvas.clear()


class TrackingData(Widget):
    flag = 0

    def state_change(self, instance, **kwargs):

        flag = kwargs['state']

        if instance.state == 'down':
            self.flag = flag

    def keyboard_press(self, state, flag):

        if state == 'down':
            self.flag = flag
        else:
            self.flag = 0

    def on_touch_down(self, touch):

        if self.flag != 0:
            xpos = (touch.pos[0] - self.pos[0]) / self.size[0]
            ypos = (touch.pos[1] - self.pos[1]) / self.size[1]
            self.parent.parent.track_ammend(np.asarray([xpos, ypos]), self.flag)
            self.flag = 0


class Jump(Widget):
    flag = 0

    def jump(self, instance):

        if instance.state == 'down':
            self.flag = 1

    def k_jump(self, state):

        if state == 'down':
            self.flag = 1
        else:
            self.flag = 0

    def on_touch_down(self, touch):

        if self.flag != 0:
            self.flag = 0
            xpos = (touch.pos[0] - self.pos[0]) / self.size[0]
            self.parent.parent.tracking_frame([], np.asarray([xpos]))


class TrackingUI(Widget):
    def track_frame_update(self):

        if self.track_ids.any:
            self.cell_mark.remove_dot()
            mask = np.intersect1d(self.track_ids, self.frame_feats[:, 0])

            if mask:
                cell_center = self.frame_feats[self.frame_feats[:, 0] == mask[0], [2, 3]]
                self.cell_mark.draw_dot(cell_center, self.dims, 1., 1., 1., 50)

        for i in range(len(self.parent.fov['tracks_stored'])):
            self.store_layout.children[i].remove_dot()

        count = 0

        if len(self.store_ids) > 0:

            mask2 = self.store_ids.intersection(set(self.frame_feats[:, 0]))
            if len(mask2) > 0:
                for i in mask2:
                    cell_center = self.frame_feats[self.frame_feats[:, 0] == i, [2, 3]]
                    self.store_layout.children[count].draw_dot(cell_center, self.dims, 0., 0., 0., 80)
                    count += 1

    def tracking_frame(self, instance, val):

        if val >= self.frames:
            val = self.frames - 1

        if val < 0:
            val = 0

        if self.track_btn5.state == 'down':
            self.track_btn5.state = 'normal'
            val = int(val * self.frames)

        self.current_frame = int(val)
        im_temp = self.labels[int(val), :, :]
        self.sframe.value = val

        mapping = np.hstack((0, self.features[:, 18].astype(int)))
        self.track_disp.update_im(im_temp.astype(float), mapping)
        self.mov_disp.update_im(self.channel_im[int(val)])

        inds = self.features[:, 1]
        mask = inds == self.current_frame
        self.frame_feats = self.features[mask, :]

        self.track_frame_update()

    def modify_update(self):

        im_temp = self.labels[self.current_frame, :, :]

        mapping = np.hstack((0, self.features[:, 18].astype(int)))
        self.track_disp.update_im(im_temp.astype(float), mapping)

        self.canvas.ask_update()

        mask2 = self.tracks[:, 4] == self.track_ind[0]
        self.track_ids = self.tracks[mask2, 0].astype(int)
        self.track_frame_update()

        self.map_ind = self.features[self.track_ids[0] - 1, 18]

        feats_temp1 = self.features[self.track_ids - 1, self.show_feat[0]]
        feats_temp2 = self.features[self.track_ids - 1, self.show_feat[1]]
        feats_temp3 = self.features[self.track_ids - 1, self.show_feat[2]]

        feats_temp1 = feats_temp1 / max(feats_temp1)
        feats_temp2 = feats_temp2 / max(feats_temp2)
        feats_temp3 = feats_temp3 / max(feats_temp3)

        t_temp = self.features[self.track_ids - 1, 1]

        if t_temp.size > 1:

            self.graph.remove_plot(self.plot1)
            self.plot1 = SmoothLinePlot(color=[1, 0, 0, 1])
            self.plot1.points = [(t_temp[i], feats_temp1[i]) for i in range(len(feats_temp1))]
            self.graph.add_plot(self.plot1)

            self.graph.remove_plot(self.plot2)
            self.plot2 = SmoothLinePlot(color=[0, 1, 0, 1])
            self.plot2.points = [(t_temp[i], feats_temp2[i]) for i in range(len(feats_temp2))]
            self.graph.add_plot(self.plot2)

            self.graph.remove_plot(self.plot3)
            self.plot3 = SmoothLinePlot(color=[0, 0, 1, 1])
            self.plot3.points = [(t_temp[i], feats_temp3[i]) for i in range(len(feats_temp3))]
            self.graph.add_plot(self.plot3)

        else:

            self.graph.remove_plot(self.plot1)
            self.graph.remove_plot(self.plot2)
            self.graph.remove_plot(self.plot3)

    def track_ammend(self, pos, flag):

        # Calculate nearest segment

        if 0 < flag <= 5:

            pos = np.asarray([[pos[0] * self.dims[1], pos[1] * self.dims[0]]])
            d = distance.cdist(self.frame_feats[:, [2, 3]], pos)  # Calculate distance from frame segments

            sel = self.frame_feats[np.argmin(d), :]  # Choose closest segment to mouse click
            mask = self.tracks[:, 0] == sel[0]  # Test if selection is in track array and identify it

            # Select Cell

            if flag == 1:
                self.track_btn1.state = 'normal'

                if sum(mask) and min(d) < 50:
                    self.track_ind = self.tracks[mask, 4]  # Set the selected track index
                    self.modify_update()  # Display this as cell marked with black dot

            # Add segment to cell track

            if flag == 2:

                self.track_btn2.state = 'normal'

                if not (sum(mask)) and min(d) < 50:

                    frame_ids = self.tracks[self.tracks[:, 5] == self.current_frame, :]  # Get all cell tracks in frame

                    mask2 = frame_ids[:, 4] == self.track_ind  # Test if selected track already has segment in frame

                    if sum(mask2):

                        feat_id = frame_ids[mask2, 0]  # get unique id of segment in frame

                        self.features[self.features[:, 0] == feat_id, 18] = 1
                        self.features[self.features[:, 0] == sel[0], 18] = self.map_ind

                        self.tracks[self.tracks[:, 0] == feat_id, 0] = sel[0]

                    else:

                        # where to insert the track

                        i = 0

                        while self.tracks[i, 4] != self.track_ind:
                            i += 1

                        else:
                            while self.tracks[i, 5] < self.current_frame and i + 1 < self.tracks.shape[0]:
                                i += 1

                            if i + 1 == self.tracks.shape[0]:

                                self.tracks = np.vstack(
                                    (self.tracks, [sel[0], 0, 0, 0, self.track_ind, self.current_frame, 0, 0]))
                                self.features[self.features[:, 0] == sel[0], 18] = self.map_ind

                            else:

                                self.tracks = np.insert(self.tracks, i,
                                                        [sel[0], 0, 0, 0, self.track_ind, self.current_frame, 0, 0], 0)
                                self.features[self.features[:, 0] == sel[0], 18] = self.map_ind

                    self.modify_update()

            # Remove segment from cell track

            if flag == 3:

                self.track_btn3.state = 'normal'

                if sum(mask) and min(d) < 50:
                    self.features[self.features[:, 0] == sel[0], 18] = 1

                    ind = np.where(mask)
                    self.tracks = np.delete(self.tracks, ind[0][0], 0)
                    self.modify_update()

            # Swap tracks in proceeding frames

            if flag == 4:

                self.track_btn4.state = 'normal'

                if sum(mask) and min(d) < 50 and self.current_frame != 0 and self.current_frame != self.frames:
                    # rows of selected track proceeding frame

                    sel_mask = self.tracks[:, 4] == self.track_ind[0]
                    sel_track = self.tracks[sel_mask, :]

                    # rows of track to swap proceeding frame

                    swap_ind = self.tracks[mask, 4]

                    swap_mask = self.tracks[:, 4] == swap_ind
                    swap_track = self.tracks[swap_mask, :]

                    # perform swap

                    sel1 = sel_track[sel_track[:, 5] < self.current_frame, :]
                    sel2 = sel_track[sel_track[:, 5] >= self.current_frame, :]

                    swap1 = swap_track[swap_track[:, 5] < self.current_frame, :]
                    swap2 = swap_track[swap_track[:, 5] >= self.current_frame, :]

                    swapped_1 = np.vstack((sel1, swap2))
                    swapped_1[:, 4] = sel1[0, 4]

                    swapped_2 = np.vstack((swap1, sel2))
                    swapped_2[:, 4] = swap1[0, 4]

                    self.tracks = self.tracks[np.logical_not(np.logical_or(sel_mask, swap_mask)), :]
                    self.tracks = np.vstack((self.tracks, swapped_1, swapped_2))

                    # update labels

                    map_ind1 = self.features[int(swapped_1[0, 0]) - 1, 18]
                    map_ind2 = self.features[int(swapped_2[0, 0]) - 1, 18]

                    self.features[swapped_1[:, 0].astype(int) - 1, 18] = map_ind1
                    self.features[swapped_2[:, 0].astype(int) - 1, 18] = map_ind2

                    self.modify_update()

            # Create new track

            if flag == 5:

                self.track_btn6.state = 'normal'

                if not (sum(mask)) and min(d) < 50:
                    # Create new track
                    self.track_ind = np.asarray([max(self.tracks[:, 4]) + 1])

                    r = int(252 * np.random.rand()) + 3
                    self.map_ind = r

                    self.tracks = np.vstack((self.tracks, [sel[0], 0, 0, 0, self.track_ind, self.current_frame, 0, 0]))
                    self.features[self.features[:, 0] == sel[0], 18] = r

                    self.modify_update()

                    stored_temp = self.parent.fov['tracks_stored'][...]
                    stored_temp = np.append(stored_temp, 0)
                    del self.parent.fov['tracks_stored']
                    self.parent.fov.create_dataset("tracks_stored", data=stored_temp)
                    self.store_layout.add_widget(CellMark(size_hint=(.43, .43), pos_hint={'x': .12, 'y': .44}))

    def keyboard_closed(self):
        self.keyboard.unbind(on_key_down=self.key_print)
        self.keyboard = None

    def key_print(self, keyboard, keycode, text, modifiers):

        key = keycode[1]

        if key == 'a':
            self.tracking_frame([], self.current_frame - 1)
            self.canvas.ask_update()

        if key == 'd':
            self.tracking_frame([], self.current_frame + 1)
            self.canvas.ask_update()

        if key == 'z':

            if self.track_btn1.state == 'normal':
                self.track_btn1.state = 'down'

            else:
                self.track_btn1.state = 'normal'

            self.tracking_window.keyboard_press(self.track_btn1.state, 1)

        if key == 'c':

            if self.track_btn2.state == 'normal':
                self.track_btn2.state = 'down'

            else:
                self.track_btn2.state = 'normal'

            self.tracking_window.keyboard_press(self.track_btn2.state, 2)

        if key == 'v':

            if self.track_btn3.state == 'normal':
                self.track_btn3.state = 'down'

            else:
                self.track_btn3.state = 'normal'

            self.tracking_window.keyboard_press(self.track_btn3.state, 3)

        if key == 'x':

            if self.track_btn4.state == 'normal':
                self.track_btn4.state = 'down'

            else:
                self.track_btn4.state = 'normal'

            self.tracking_window.keyboard_press(self.track_btn4.state, 4)

        if key == 'n':

            if self.track_btn6.state == 'normal':
                self.track_btn6.state = 'down'

            else:
                self.track_btn6.state = 'normal'

            self.tracking_window.keyboard_press(self.track_btn6.state, 6)

        if key == 'w':

            if self.track_btn5.state == 'normal':
                self.track_btn5.state = 'down'

            else:
                self.track_btn5.state = 'normal'

            self.jump_window.k_jump(self.track_btn5.state)

    def save_tracks(self, instance):

        for g in self.parent.fov:
            if g == 'saved_tracks':
                del self.parent.fov['saved_tracks']
            if g == 'saved_feats':
                del self.parent.fov['saved_feats']

        self.parent.fov.create_dataset("saved_tracks", data=self.tracks)
        self.parent.fov.create_dataset("saved_feats", data=self.features)

    def load_tracks(self, instance):

        self.tracks = self.parent.fov['saved_tracks'][:, :]
        self.features = self.parent.fov['saved_feats'][:, :]

        im_temp = self.labels[self.current_frame, :, :]

        mapping = np.hstack((0, self.features[:, 18].astype(int)))
        self.track_disp.update_im(im_temp.astype(float), mapping)

        self.canvas.ask_update()

    def store_track(self, instance):

        if self.parent.fov['tracks_stored'][int(self.track_ind)] == 0:
            self.parent.fov['tracks_stored'][int(self.track_ind)] = 1
            self.store_ids = self.store_ids.union(set(self.tracks[self.tracks[:, 4] == self.track_ind, 0]))

        else:
            self.parent.fov['tracks_stored'][int(self.track_ind)] = 0
            self.store_ids = self.store_ids.difference(set(self.tracks[self.tracks[:, 4] == self.track_ind, 0]))

        self.modify_update()

    def save_csv(self, instance):

        ''' Create matrix and write to csv for features. Features are: Track_id, Frame, X_center, Y_center, Area,
        Eccentricity, Solidity, Perimeter, CH1 Mean Intensity, CH1 StdDev Intensity, CH1 Floored Mean, CH2 Mean Intensity,
        CH2 StdDev Intensity, CH3 Mean Intensity, CH3 StdDev Intensity, '''

        feat_mat = np.zeros((1, 18))

        for i in range(1, int(max(self.tracks[:, 4])) + 1):
            if sum(self.tracks[:, 4] == i) > 0:

                track_temp = self.tracks[self.tracks[:, 4] == i, :]

                for j in range(track_temp.shape[0]):
                    mask = self.features[:, 0] == track_temp[j, 0]
                    fv = self.features[mask, :]
                    feat_mat = np.vstack((feat_mat,
                                          [i, track_temp[j, 5], fv[0, 2], fv[0, 3], fv[0, 5], fv[0, 6], fv[0, 7],
                                           fv[0, 8], fv[0, 9], fv[0, 10],
                                           fv[0, 11], fv[0, 20], fv[0, 21], fv[0, 22], fv[0, 23], track_temp[j, 3],
                                           track_temp[j, 0], 0]))

        feat_mat = np.delete(feat_mat, 0, 0)

        for i in range(feat_mat.shape[0]):

            if feat_mat[i, 15] > 0:

                mask = feat_mat[:, 16] == feat_mat[i, 15]
                ind_change = feat_mat[mask, 0]

                frame_change = feat_mat[mask, 1]
                mask_change = np.logical_and(feat_mat[:, 0] == ind_change, feat_mat[:, 1] > frame_change)
                if sum(mask_change) > 0:
                    # feat_mat[mask_change, 0] = max(feat_mat[:, 0]) + 1  #option to change index of parent track daughter cell

                    change_list = np.where(mask_change)
                    feat_mat[change_list[0][0], 17] = ind_change
                    feat_mat[i, 17] = ind_change

        with open('Results.csv', 'wb') as f:
            f.write(b'Track ID, Frame, X center, Y center, Area, Eccentricity, Solidity, Perimeter, '
                    b'CH1 Mean Intensity, CH1 StdDev Intensity, CH1 Floored Mean, CH2 Mean Intensity, '
                    b'CH2 StdDev Intensity, CH3 Mean Intensity, CH3 StdDev Intensity, Parent Track ID\n')

            feat_mat2 = np.delete(feat_mat, [15, 16], 1)
            np.savetxt(f, feat_mat2, delimiter=",")

        '''
        ### Data formatting for iscb benchmark dataset

        for i in range(feat_mat.shape[0]):

            if feat_mat[i,15] > 0:

                mask = feat_mat[:, 16] == feat_mat[i, 15]
                ind_change = feat_mat[mask, 0]

                frame_change = feat_mat[mask, 1]
                mask_change = np.logical_and(feat_mat[:,0] == ind_change, feat_mat[:,1] > frame_change)
                if sum(mask_change)>0:
                    feat_mat[mask_change, 0] = max(feat_mat[:, 0]) + 1
                    change_list = np.where(mask_change)
                    feat_mat[change_list[0][0], 17] = ind_change
                    feat_mat[i, 17] = ind_change

        np.savetxt("Results2.csv", feat_mat, delimiter=",")

        track_text = np.zeros((int(max(feat_mat[:,0])), 4))

        for i in range(1, int(max(feat_mat[:,0])+1)):

            track_temp = feat_mat[feat_mat[:,0]==i,:]
            track_text[i-1, 0] = i
            track_text[i-1, 1] = track_temp[0,1]
            track_text[i-1, 2] = track_temp[-1,1]
            track_text[i-1, 3] = track_temp[0,17]

        np.savetxt("res_track.txt", track_text.astype(int),fmt='%i', delimiter=" ")

        for i in range(self.frames):

            im_temp = self.labels[i, :, :]
            im_tracked = np.zeros(im_temp.shape)

            l_temp = feat_mat[feat_mat[:, 1] == i,:]

            for j in range(l_temp.shape[0]):
                ind_seg = im_temp[int(l_temp[j, 3]), int(l_temp[j, 2])]
                im_tracked[im_temp == ind_seg] = l_temp[j,0]

            n = str(i)
            n = n.zfill(2)
            n = 'mask' + n + '.tif'
            im_tracked = im_tracked.astype(np.uint16)
            tifffile.imsave(n, im_tracked)
        '''

    def save_sel_csv(self, instance):

        feat_mat = np.zeros((1, 18))

        for i in range(1, int(max(self.tracks[:, 4]))):
            if self.parent.fov['tracks_stored'][i] == 1:
                if sum(self.tracks[:, 4] == i) > 0:

                    track_temp = self.tracks[self.tracks[:, 4] == i, :]

                    for j in range(track_temp.shape[0]):
                        mask = self.features[:, 0] == track_temp[j, 0]
                        fv = self.features[mask, :]
                        feat_mat = np.vstack((feat_mat,
                                              [i, track_temp[j, 5], fv[0, 2], fv[0, 3], fv[0, 5], fv[0, 6], fv[0, 7],
                                               fv[0, 8], fv[0, 9], fv[0, 10],
                                               fv[0, 11], fv[0, 20], fv[0, 21], fv[0, 22], fv[0, 23], track_temp[j, 3],
                                               track_temp[j, 0], 0]))

        feat_mat = np.delete(feat_mat, 0, 0)

        for i in range(feat_mat.shape[0]):

            if feat_mat[i, 15] > 0:

                mask = feat_mat[:, 16] == feat_mat[i, 15]
                ind_change = feat_mat[mask, 0]

                frame_change = feat_mat[mask, 1]
                mask_change = np.logical_and(feat_mat[:, 0] == ind_change, feat_mat[:, 1] > frame_change)
                if sum(mask_change) > 0:
                    # feat_mat[mask_change, 0] = max(feat_mat[:, 0]) + 1  #option to change index of parent track daughter cell

                    change_list = np.where(mask_change)
                    feat_mat[change_list[0][0], 17] = ind_change
                    feat_mat[i, 17] = ind_change

        with open('Results.csv', 'wb') as f:

            f.write(b'Track ID, Frame, X center, Y center, Area, Eccentricity, Solidity, Perimeter, '
                    b'CH1 Mean Intensity, CH1 StdDev Intensity, CH1 Floored Mean, CH2 Mean Intensity, '
                    b'CH2 StdDev Intensity, CH3 Mean Intensity, CH3 StdDev Intensity, Parent Track ID\n')

            feat_mat2 = np.delete(feat_mat, [15, 16], 1)
            np.savetxt(f, feat_mat2, delimiter=",")

    def feat1(self, instance):
        num = int(''.join([instance.text]))

        if num >= 0 and num < len(self.feat_inds):
            self.show_feat[0] = self.feat_inds[num]

        self.modify_update()

    def feat2(self, instance):
        num = int(''.join([instance.text]))

        if num >= 0 and num < len(self.feat_inds):
            self.show_feat[1] = self.feat_inds[num]

        self.modify_update()

    def feat3(self, instance):
        num = int(''.join([instance.text]))

        if num >= 0 and num < len(self.feat_inds):
            self.show_feat[2] = self.feat_inds[num]

        self.modify_update()

    def initialize(self, channel_im, labels, frames):

        self.tracks = self.parent.fov['tracks'][:, :]
        self.features = self.parent.fov['feats'][:, :]

        self.keyboard = Window.request_keyboard(self.keyboard_closed, self)
        self.keyboard.bind(on_key_down=self.key_print)

        self.labels = labels
        self.channel_im = channel_im
        self.frames = frames

        self.dims = self.channel_im[0].shape
        self.tr_layout = FloatLayout(size=(Window.width, Window.height))

        # Stored Tracks

        self.store_ids = set()

        for i in range(len(self.parent.fov['tracks_stored'])):
            if self.parent.fov['tracks_stored'][i] == 1:
                self.store_ids = self.store_ids.union(set(self.tracks[self.tracks[:, 4] == i, 0]))

        self.store_layout = FloatLayout(size=(Window.width, Window.height))

        self.track_disp = IndexedDisplay(size_hint=(.43, .43), pos_hint={'x': .12, 'y': .44})
        self.tr_layout.add_widget(self.track_disp)

        self.mov_disp = ImDisplay(size_hint=(.43, .43), pos_hint={'x': .56, 'y': .44})
        self.tr_layout.add_widget(self.mov_disp)

        self.track_ids = np.zeros(1)
        self.current_frame = 0

        inds = self.features[:, 1]
        mask = inds == 0

        if sum(mask.astype(int)) > 0:
            self.frame_feats = self.features[mask, :]

        im_temp = self.labels[0, :, :]

        mapping = np.hstack((0, self.features[:, 18].astype(int)))
        self.track_disp.create_im(im_temp, 'Random', mapping)

        self.mov_disp.create_im(self.channel_im[0], 'PastelHeat')

        self.sframe = Slider(min=0, max=self.frames - 1, value=1, size_hint=(.3, .1), pos_hint={'x': .2, 'y': .9})
        self.tr_layout.add_widget(self.sframe)
        self.sframe.bind(value=self.tracking_frame)

        self.tracking_window = TrackingData(size_hint=(.43, .43), pos_hint={'x': .12, 'y': .44})
        self.jump_window = Jump(size_hint=(.87, .3), pos_hint={'x': .12, 'y': .12})

        layout4 = GridLayout(cols=1, padding=2, size_hint=(.1, .8), pos_hint={'x': .01, 'y': .1})
        self.cell_mark = CellMark(size_hint=(.43, .43), pos_hint={'x': .12, 'y': .44})

        self.track_btn1 = ToggleButton(text='Select\n Cell (z)')
        self.track_btn2 = ToggleButton(text=' Add \nSegment (c)')
        self.track_btn3 = ToggleButton(text='Remove\nSegment(v)')
        self.track_btn4 = ToggleButton(text='Swap\n Tracks(x)')
        self.track_btn5 = ToggleButton(text='Jump(w)')
        self.track_btn6 = ToggleButton(text='New \n Track(n)')
        self.track_btn9 = Button(text='Store \n Track')

        self.track_btn7 = Button(text='Save \n Tracks')
        self.track_btn8 = Button(text='Load \n Tracks')

        self.track_btn10 = Button(text='Export all\n to CSV(n)')
        self.track_btn11 = Button(text='Export sel\n to CSV(n)')

        self.track_btn1.bind(on_press=partial(self.tracking_window.state_change, state=1))
        self.track_btn2.bind(on_press=partial(self.tracking_window.state_change, state=2))
        self.track_btn3.bind(on_press=partial(self.tracking_window.state_change, state=3))
        self.track_btn4.bind(on_press=partial(self.tracking_window.state_change, state=4))
        self.track_btn5.bind(on_press=self.jump_window.jump)
        self.track_btn6.bind(on_press=partial(self.tracking_window.state_change, state=5))

        self.track_btn7.bind(on_press=self.save_tracks)
        self.track_btn8.bind(on_press=self.load_tracks)
        self.track_btn9.bind(on_press=self.store_track)

        self.track_btn10.bind(on_press=self.save_csv)
        self.track_btn11.bind(on_press=self.save_sel_csv)

        layout4.add_widget(self.track_btn1)
        layout4.add_widget(self.track_btn2)
        layout4.add_widget(self.track_btn3)
        layout4.add_widget(self.track_btn4)
        layout4.add_widget(self.track_btn6)
        layout4.add_widget(self.track_btn5)
        layout4.add_widget(self.track_btn7)
        layout4.add_widget(self.track_btn8)
        layout4.add_widget(self.track_btn9)
        layout4.add_widget(self.track_btn10)
        layout4.add_widget(self.track_btn11)

        self.layout4 = layout4

        self.feat_inds = [5, 6, 7, 8, 9, 10, 11, 20, 21, 22, 23]
        self.show_feat = [self.feat_inds[0], self.feat_inds[4], self.feat_inds[6]]

        self.graph = Graph(background_color=[1., 1., 1., 1.], draw_border=False,
                           xmax=self.frames, ymin=0,
                           ymax=1,
                           size_hint=(.87, .3), pos_hint={'x': .12, 'y': .12})

        self.plot1 = SmoothLinePlot(color=[1, 0, 0, 1])
        self.plot1.points = [(0, 0), (0, 0)]
        self.graph.add_plot(self.plot1)

        self.plot2 = SmoothLinePlot(color=[0, 1, 0, 1])
        self.plot2.points = [(0, 0), (0, 0)]
        self.graph.add_plot(self.plot2)

        self.plot3 = SmoothLinePlot(color=[0, 0, 1, 1])
        self.plot3.points = [(0, 0), (0, 0)]
        self.graph.add_plot(self.plot3)

        self.text_input1 = TextInput(text='Feat1',
                                     multiline=False, size_hint=(.09, .05), pos_hint={'x': .7, 'y': .12})
        self.text_input2 = TextInput(text='Feat2',
                                     multiline=False, size_hint=(.09, .05), pos_hint={'x': .8, 'y': .12})
        self.text_input3 = TextInput(text='Feat3',
                                     multiline=False, size_hint=(.09, .05), pos_hint={'x': .9, 'y': .12})

        self.text_input1.bind(on_text_validate=self.feat1)
        self.text_input2.bind(on_text_validate=self.feat2)
        self.text_input3.bind(on_text_validate=self.feat3)

        with self.canvas:
            self.add_widget(self.tr_layout)
            self.add_widget(self.store_layout)

            self.tr_layout.add_widget(self.layout4)
            self.tr_layout.add_widget(self.tracking_window)
            self.tr_layout.add_widget(self.jump_window)
            self.tr_layout.add_widget(self.cell_mark)
            self.tr_layout.add_widget(self.graph)

            self.tr_layout.add_widget(self.text_input1)
            self.tr_layout.add_widget(self.text_input2)
            self.tr_layout.add_widget(self.text_input3)

            for i in range(len(self.parent.fov['tracks_stored'])):
                self.store_layout.add_widget(CellMark(size_hint=(.43, .43), pos_hint={'x': .12, 'y': .44}))

    def remove(self):

        self.tr_layout.clear_widgets()
        self.store_layout.clear_widgets()

    def update_size(self, window, width, height):
        self.tr_layout.width = width
        self.tr_layout.height = height

        self.store_layout.width = width
        self.store_layout.height = height
