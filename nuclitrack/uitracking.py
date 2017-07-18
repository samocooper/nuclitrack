import numpy as np

from kivy.graphics import Ellipse, Color
from kivy.uix.widget import Widget
from kivy.uix.slider import Slider
from kivy.uix.togglebutton import ToggleButton, Button
from kivy.uix.gridlayout import GridLayout
from kivy.core.window import Window
from kivy.uix.floatlayout import FloatLayout
from functools import partial
from kivy.uix.textinput import TextInput
from kivy.uix.label import Label
from scipy.spatial import distance
from kivy.uix.dropdown import DropDown

from .imagewidget import ImDisplay, IndexedDisplay
from .graph import Graph, SmoothLinePlot
from . import trackcells
from PIL import Image

class RunTracking(Widget):

    def __init__(self, features, track_param, frames, **kwargs):

        super().__init__(**kwargs)

        self.tracking_object = trackcells.TrackCells(features=features, track_param=track_param, frames=frames)

        self.track_message = Label(text='[b][color=000000] Tracking cells [/b][/color]', markup=True,
                                   size_hint=(.2, .05), pos_hint={'x': .4, 'y': .65})
        self.track_counter = Label(text='[b][color=000000] [/b][/color]', markup=True,
                                   size_hint=(.2, .05), pos_hint={'x': .4, 'y': .6})
        self.cancel_btn = Button(text='Cancel', markup=True,
                                   size_hint=(.2, .05), pos_hint={'x': .4, 'y': .5})
        self.cancel_btn.bind(on_release=self.cancel_tracking)
        self.layout = FloatLayout(size=(Window.width, Window.height))

        self.add_flag = True
        self.optimise_flag = False
        self.count = 0
        self.optimise_count = 0
        self.sweep = 0
        self.cancel_flag = False

        with self.canvas:
            self.add_widget(self.layout)
            self.layout.add_widget(self.track_counter)
            self.layout.add_widget(self.track_message)
            self.layout.add_widget(self.cancel_btn)

    def cancel_tracking(self, instance):

        self.add_flag = False
        self.optimise_flag = False
        self.cancel_flag = True
        self.parent.cancel_flag = True
        self.cancel_btn.text = 'Tracking Canceled'

    def update_count(self):
        if self.add_flag:
            self.track_counter.text = '[b][color=000000]' + str(
                self.count) + '[/b][/color]'
        else:
            self.track_counter.text = '[b][color=000000]' + str(self.optimise_count) + '[/b][/color]'

    def update_message(self, val):

        if val >= 0:
            self.track_message.text = '[b][color=000000]Optimising Tracks Sweep ' + str(
                val) + ' [/b][/color]'
            self.tracking_state = val

        else:
            self.track_message.text = '[b][color=000000]Tracking Completed | Total segments: ' + \
                                              str(self.segment_count) + ' | Total double segments: ' + \
                                              str(self.double_segment) + '[/b][/color]'

    def add_track(self):

        if self.add_flag:

            self.add_flag = self.tracking_object.addtrack()
            self.count += 1

            if not self.add_flag:

                self.optimise_flag = True
                self.update_message(self.sweep+1)

            return False

        if self.optimise_flag:

            self.optimise_flag = self.tracking_object.optimisetrack()
            self.optimise_count += 1

            if not self.optimise_flag and self.sweep < 1:

                self.optimise_count = 0
                self.sweep += 1
                self.update_message(self.sweep + 1)
                self.optimise_flag = True

            return False

        if not self.add_flag and not self.optimise_flag:
            if not self.cancel_flag:

                self.optimise_count = 0
                self.tracks, self.features, self.segment_count, self.double_segment = self.tracking_object.get()
                self.update_message(-1)

                return True

    def test_cancel(self):
        return self.cancel_flag

    def get(self):
        return self.tracks, self.features


class CellMark(Widget):

    def draw_dot(self, cell_center, dims, r, g, b, d):

        self.d = d
        cell_center[0] = cell_center[0] / dims[1]
        cell_center[1] = (1 - cell_center[1] / dims[0])

        self.cell_center = cell_center

        ds = (self.d / self.height)*100
        xpos = (cell_center[0] * self.width) + self.pos[0] - ds / 2
        ypos = (cell_center[1] * self.height) + self.pos[1] - ds / 2

        with self.canvas:
            Color(r, g, b)
            self.dot = Ellipse(size=(ds, ds), pos=(xpos, ypos))

        self.bind(pos=self.update_im, size=self.update_im)

    def update_im(self, *args):

        ds = (self.d / self.height)*100
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
            ypos = (1 - (touch.pos[1] - self.pos[1]) / self.size[1])
            self.parent.parent.track_ammend(np.asarray([xpos, ypos]), self.flag)
            self.flag = 0


class Jump(Widget):
    is_down = 0
    flag = 0

    def jump(self, instance, flag):

        if instance.state == 'down':
            self.is_down = 1
            self.flag = flag

    def k_jump(self, state, flag):

        if state == 'down':
            self.is_down = 1
            self.flag = flag
        else:
            self.is_down = 0

    def on_touch_down(self, touch):

        if self.is_down != 0:
            self.is_down = 0
            xpos = (touch.pos[0] - self.pos[0]) / self.size[0]
            if self.flag == 0:
                self.parent.parent.frame_select([], np.asarray([xpos]))
            else:
                self.parent.parent.add_event(xpos, self.flag)


class TrackingUI(Widget):

    def __init__(self, file_list, labels, tracks, stored_tracks, features, frames,
                 dims, channels, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.file_list = file_list
        self.channels = channels
        self.tracks = tracks
        self.features = features
        self.dims = dims
        self.labels = labels

        self.keyboard = Window.request_keyboard(self.keyboard_closed, self)
        self.keyboard.bind(on_key_down=self.key_print)

        self.frames = frames

        self.tr_layout = FloatLayout(size=(Window.width, Window.height))

        # Stored Tracks

        self.store_ids = set()

        for i in range(len(stored_tracks)):
            if stored_tracks[i] == 1:
                self.store_ids = self.store_ids.union(set(self.tracks[self.tracks[:, 4] == i, 0]))

        self.store_layout = FloatLayout(size=(Window.width, Window.height))

        self.track_disp = IndexedDisplay(size_hint=(.43, .43), pos_hint={'x': .12, 'y': .46})
        self.tr_layout.add_widget(self.track_disp)

        self.mov_disp = ImDisplay(size_hint=(.43, .43), pos_hint={'x': .56, 'y': .46})
        self.tr_layout.add_widget(self.mov_disp)

        self.track_ids = np.zeros(1)
        self.current_frame = 0

        inds = self.features['tracking'][:, 1]
        mask = inds == 0

        if sum(mask.astype(int)) > 0:
            self.frame_feats = self.features['tracking'][mask, :]

        im_temp = self.labels[0, :, :]

        mapping = self.features['tracking'][:, 11].astype(int)
        self.track_disp.create_im(im_temp, 'Random', mapping)

        self.channel = 0
        im = np.asarray(Image.open(self.file_list[self.channel][0]))
        im = im.astype(float)
        self.mov_disp.create_im(im, 'PastelHeat')

        self.frame_slider = Slider(min=0, max=self.frames - 1, value=1, size_hint=(.3, .06),
                                   pos_hint={'x': .145, 'y': .94})
        self.frame_slider.bind(value=self.frame_select)

        self.frame_text = Label(text='[color=000000]Frame: ' + str(0) + '[/color]',
                                size_hint=(.15, .04), pos_hint={'x': .22, 'y': .9}, markup=True)
        self.frame_minus = Button(text='<<:a',
                                  size_hint=(.07, .03), pos_hint={'x': .145, 'y': .905}, markup=True)
        self.frame_plus = Button(text='d:>>',
                                 size_hint=(.07, .03), pos_hint={'x': .375, 'y': .905}, markup=True)
        self.frame_minus.bind(on_press=self.frame_backward)
        self.frame_plus.bind(on_press=self.frame_forward)

        self.tr_layout.add_widget(self.frame_slider)
        self.tr_layout.add_widget(self.frame_plus)
        self.tr_layout.add_widget(self.frame_minus)
        self.tr_layout.add_widget(self.frame_text)

        self.tracking_window = TrackingData(size_hint=(.43, .43), pos_hint={'x': .12, 'y': .46})
        self.jump_window = Jump(size_hint=(.87, .3), pos_hint={'x': .12, 'y': .12})

        layout4 = GridLayout(cols=1, padding=2, size_hint=(.1, .78), pos_hint={'x': .01, 'y': .115})

        self.cell_mark = CellMark(size_hint=(.43, .43), pos_hint={'x': .12, 'y': .46})
        self.cell_mark_2 = CellMark(size_hint=(.43, .43), pos_hint={'x': .56, 'y': .46})

        self.track_btn1 = ToggleButton(text=' Select Cell (z) ', markup=True, halign='center', valign='middle')
        self.track_btn2 = ToggleButton(text=' Add Segment (c)', markup=True, halign='center', valign='middle')
        self.track_btn3 = ToggleButton(text='Remove Segment(v)', markup=True, halign='center', valign='middle')
        self.track_btn4 = ToggleButton(text='Swap Tracks (x)', markup=True, halign='center', valign='middle')
        self.track_btn5 = ToggleButton(text='Jump (w)', markup=True, halign='center', valign='middle')
        self.track_btn6 = ToggleButton(text='New Track (n)', markup=True, halign='center', valign='middle')
        self.track_btn9 = Button(text='Store Track', markup=True, halign='center', valign='middle')
        self.track_btn10 = Button(text='Export All to CSV', markup=True, halign='center', valign='middle')
        self.track_btn11 = Button(text='Export Sel to CSV', markup=True, halign='center', valign='middle')

        self.track_btn1.bind(on_press=partial(self.tracking_window.state_change, state=1))
        self.track_btn2.bind(on_press=partial(self.tracking_window.state_change, state=2))
        self.track_btn3.bind(on_press=partial(self.tracking_window.state_change, state=3))
        self.track_btn4.bind(on_press=partial(self.tracking_window.state_change, state=4))
        self.track_btn5.bind(on_press=partial(self.jump_window.jump, flag=0))
        self.track_btn6.bind(on_press=partial(self.tracking_window.state_change, state=5))
        self.track_btn9.bind(on_press=self.store_track)
        self.track_btn10.bind(on_press=self.save_csv)
        self.track_btn11.bind(on_press=self.save_sel_csv)

        layout4.add_widget(self.track_btn1)
        layout4.add_widget(self.track_btn2)
        layout4.add_widget(self.track_btn3)
        layout4.add_widget(self.track_btn4)
        layout4.add_widget(self.track_btn6)
        layout4.add_widget(self.track_btn5)
        layout4.add_widget(self.track_btn9)
        layout4.add_widget(self.track_btn10)
        layout4.add_widget(self.track_btn11)

        for child in layout4.children:
            child.bind(size=child.setter('text_size'))

        self.layout4 = layout4

        self.show_feat = [0, 1, 2]

        self.graph = Graph(background_color=[1., 1., 1., 1.], draw_border=False,
                           xmax=self.frames, ymin=0,
                           ymax=1,
                           size_hint=(.87, .32), pos_hint={'x': .12, 'y': .12})

        self.plot1 = SmoothLinePlot(color=[1, 0, 0, 1])
        self.plot1.points = [(0, 0), (0, 0)]
        self.graph.add_plot(self.plot1)

        self.plot2 = SmoothLinePlot(color=[0, 1, 0, 1])
        self.plot2.points = [(0, 0), (0, 0)]
        self.graph.add_plot(self.plot2)

        self.plot3 = SmoothLinePlot(color=[0, 0, 1, 1])
        self.plot3.points = [(0, 0), (0, 0)]
        self.graph.add_plot(self.plot3)

        self.plot4 = SmoothLinePlot(color=[1, 1, 0, 1])
        self.plot4.points = [(0, 0), (0, 0)]
        self.graph.add_plot(self.plot4)

        self.plot5 = SmoothLinePlot(color=[0, 1, 1, 1])
        self.plot5.points = [(0,0), (0,0)]
        self.graph.add_plot(self.plot5)

        self.plot6 = SmoothLinePlot(color=[1, 0, 1, 1])
        self.plot6.points = [(0, 0), (0, 0)]
        self.graph.add_plot(self.plot6)

        self.plotT = SmoothLinePlot(color=[0, 0, 0, 1])
        self.plotT.points = [(0, -1), (0, 1), (0, -1)]
        self.graph.add_plot(self.plotT)

        self.text_input1 = TextInput(text='1',
                                     multiline=False, size_hint=(.08, .04), pos_hint={'x': .555, 'y': .94})
        self.text_input2 = TextInput(text='2',
                                     multiline=False, size_hint=(.08, .04), pos_hint={'x': .64, 'y': .94})
        self.text_input3 = TextInput(text='3',
                                     multiline=False, size_hint=(.08, .04), pos_hint={'x': .725, 'y': .94})

        self.text_input1.bind(on_text_validate=partial(self.feat_change, 0))
        self.text_input2.bind(on_text_validate=partial(self.feat_change, 1))
        self.text_input3.bind(on_text_validate=partial(self.feat_change, 2))

        self.event_flag1 = ToggleButton(text='Event 1', size_hint=(.08, .034), pos_hint={'x': .555, 'y': .905})
        self.event_flag2 = ToggleButton(text='Event 2', size_hint=(.08, .034), pos_hint={'x': .64, 'y': .905})
        self.event_flag3 = ToggleButton(text='Event 3', size_hint=(.08, .034), pos_hint={'x': .725, 'y': .905})

        self.event_flag1.bind(on_press=partial(self.jump_window.jump, flag=1))
        self.event_flag2.bind(on_press=partial(self.jump_window.jump, flag=2))
        self.event_flag3.bind(on_press=partial(self.jump_window.jump, flag=3))

        # Drop down menu for choosing which channel
        self.channel_choice = DropDown()

        for i in range(self.channels):
            channel_btn = ToggleButton(text='Channel ' + str(i + 1), group='channel', size_hint_y=None)
            channel_btn.bind(on_press=partial(self.change_channel, i))
            self.channel_choice.add_widget(channel_btn)

        self.main_button = Button(text=' Channel ', size_hint=(.15, .04),
                                  pos_hint={'x': .83, 'y': .94}, markup=True)
        self.main_button.bind(on_release=self.channel_choice.open)
        self.channel_choice.bind(on_select=lambda instance, x: setattr(self.main_button, 'text', x))
        self.tr_layout.add_widget(self.main_button)

        self.clear_button = Button(text=' Clear Events ', size_hint=(.15, .035),
                                  pos_hint={'x': .83, 'y': .905}, markup=True)
        self.clear_button.bind(on_press=self.clear_events)

        mask = self.tracks[:, 0] == self.tracks[0, 0]  # Test if selection is in track array and identify it
        self.track_ind = self.tracks[mask, 4]  # Set the selected track index

        with self.canvas:

            self.add_widget(self.tr_layout)
            self.add_widget(self.store_layout)

            self.tr_layout.add_widget(self.layout4)
            self.tr_layout.add_widget(self.tracking_window)
            self.tr_layout.add_widget(self.jump_window)
            self.tr_layout.add_widget(self.cell_mark)
            self.tr_layout.add_widget(self.cell_mark_2)

            self.tr_layout.add_widget(self.graph)

            self.tr_layout.add_widget(self.text_input1)
            self.tr_layout.add_widget(self.text_input2)
            self.tr_layout.add_widget(self.text_input3)

            self.tr_layout.add_widget(self.event_flag1)
            self.tr_layout.add_widget(self.event_flag2)
            self.tr_layout.add_widget(self.event_flag3)
            self.tr_layout.add_widget(self.clear_button)

            for i in range(len(stored_tracks)):
                self.store_layout.add_widget(CellMark(size_hint=(.43, .43), pos_hint={'x': .12, 'y': .46}))

    def frame_forward(self, instance):
        if self.frame_slider.value < self.frames - 1:
            self.frame_slider.value += 1

    def frame_backward(self, instance):
        if self.frame_slider.value > 0:
            self.frame_slider.value -= 1

    def track_frame_update(self):

        if self.track_ids.any:
            self.cell_mark.remove_dot()
            self.cell_mark_2.remove_dot()

            mask = np.intersect1d(self.track_ids, self.frame_feats[:, 0])

            if mask:

                cell_center = self.frame_feats[self.frame_feats[:, 0] == mask[0], [2, 3]]

                self.cell_mark.draw_dot(cell_center.copy(), self.dims, 1., 1., 1., 15)
                self.cell_mark_2.draw_dot(cell_center.copy(), self.dims, 1., 0., 0., 12)

        for i in range(len(self.parent.fov['tracks_stored'])):
            self.store_layout.children[i].remove_dot()

        count = 0

        if len(self.store_ids) > 0:

            mask2 = self.store_ids.intersection(set(self.frame_feats[:, 0]))
            if len(mask2) > 0:
                for i in mask2:
                    cell_center = self.frame_feats[self.frame_feats[:, 0] == i, [2, 3]]
                    self.store_layout.children[count].draw_dot(cell_center, self.dims, 0., 0., 0., 12)
                    count += 1


    def frame_select(self, instance, val):

        if val >= self.frames:
            val = self.frames - 1

        if val < 0:
            val = 0

        if self.track_btn5.state == 'down':
            self.track_btn5.state = 'normal'
            val = int(val * self.frames)

        self.current_frame = int(val)
        im_temp = self.labels[int(val), :, :]
        self.frame_slider.value = val

        mapping = self.features['tracking'][:, 11].astype(int)
        self.track_disp.update_im(im_temp.astype(float), mapping)

        im = np.asarray(Image.open(self.file_list[self.channel][int(val)]))
        im = im.astype(float)
        self.mov_disp.update_im(im)

        inds = self.features['tracking'][:, 1]
        mask = inds == self.current_frame
        self.frame_feats = self.features['tracking'][mask, :]

        self.frame_text.text = '[color=000000]Frame: ' + str(int(val)) + '[/color]'
        self.track_frame_update()

        self.graph.remove_plot(self.plotT)
        self.plotT = SmoothLinePlot(color=[0, 0, 0, 1])
        self.plotT.points = [(val, -1), (val, 1), (val, -1)]
        self.graph.add_plot(self.plotT)

    def clear_events(self, instance):

        mask2 = self.tracks[:, 4] == self.track_ind[0]
        self.track_ids = self.tracks[mask2, 0].astype(int)
        self.features['tracking'][self.track_ids, 12] = 0

        self.modify_update()


    def add_event(self, xpos, val):

        frame = np.round(self.frames*xpos)
        mask2 = self.tracks[:, 4] == self.track_ind[0]
        self.track_ids = self.tracks[mask2, 0].astype(int)

        time_points = self.features['tracking'][self.track_ids, 1]
        event_ind = np.argmin(np.abs(time_points - frame))
        event_ind = self.track_ids[event_ind]

        if val == 1:
            self.event_flag1.state = 'normal'
            self.features['tracking'][event_ind, 12] = 1

        if val == 2:
            self.event_flag2.state = 'normal'
            self.features['tracking'][event_ind, 12] = 2

        if val == 3:
            self.event_flag3.state = 'normal'
            self.features['tracking'][event_ind, 12] = 3

        self.modify_update()

    def modify_update(self):

        im_temp = self.labels[self.current_frame, :, :]

        mapping = self.features['tracking'][:, 11].astype(int)
        self.track_disp.update_im(im_temp.astype(float), mapping)

        self.canvas.ask_update()

        mask2 = self.tracks[:, 4] == self.track_ind[0]
        self.track_ids = self.tracks[mask2, 0].astype(int)
        self.track_frame_update()

        self.map_ind = self.features['tracking'][self.track_ids[0], 11]

        feats_temp1 = self.features['data'][self.track_ids, self.show_feat[0]]
        feats_temp2 = self.features['data'][self.track_ids, self.show_feat[1]]
        feats_temp3 = self.features['data'][self.track_ids, self.show_feat[2]]

        feats_temp1 = feats_temp1 / np.max(feats_temp1)
        feats_temp2 = feats_temp2 / np.max(feats_temp2)
        feats_temp3 = feats_temp3 / np.max(feats_temp3)

        t_temp = self.features['tracking'][self.track_ids, 1]
        events_feat = self.features['tracking'][self.track_ids, 12]

        events_point1 = []
        events_point2 = []
        events_point3 = []

        if t_temp.size > 1:

            for i in range(len(events_feat)):

                if events_feat[i] == 1:
                    events_point1.append((t_temp[i], -1))
                    events_point1.append((t_temp[i], 1))
                    events_point1.append((t_temp[i], -1))

                if events_feat[i] == 2:
                    events_point2.append((t_temp[i], -1))
                    events_point2.append((t_temp[i], 1))
                    events_point2.append((t_temp[i], -1))

                if events_feat[i] == 3:
                    events_point3.append((t_temp[i], -1))
                    events_point3.append((t_temp[i], 1))
                    events_point3.append((t_temp[i], -1))


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

            self.graph.remove_plot(self.plot4)
            self.plot4 = SmoothLinePlot(color=[1, 1, 0, 1])
            self.plot4.points = events_point1
            self.graph.add_plot(self.plot4)

            self.graph.remove_plot(self.plot5)
            self.plot5 = SmoothLinePlot(color=[0, 1, 1, 1])
            self.plot5.points = events_point2
            self.graph.add_plot(self.plot5)

            self.graph.remove_plot(self.plot6)
            self.plot6 = SmoothLinePlot(color=[1, 0, 1, 1])
            self.plot6.points = events_point3
            self.graph.add_plot(self.plot6)

        else:

            self.graph.remove_plot(self.plot1)
            self.graph.remove_plot(self.plot2)
            self.graph.remove_plot(self.plot3)
            self.graph.remove_plot(self.plot4)
            self.graph.remove_plot(self.plot5)
            self.graph.remove_plot(self.plot6)

    def track_ammend(self, pos, flag):

        mod_flag = False

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

                        self.features['tracking'][self.features['tracking'][:, 0] == feat_id, 11] = 1
                        self.features['tracking'][self.features['tracking'][:, 0] == sel[0], 11] = self.map_ind

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
                                self.features['tracking'][self.features['tracking'][:, 0] == sel[0], 11] = self.map_ind

                            else:

                                self.tracks = np.insert(self.tracks, i,
                                                        [sel[0], 0, 0, 0, self.track_ind, self.current_frame, 0, 0], 0)
                                self.features['tracking'][self.features['tracking'][:, 0] == sel[0], 11] = self.map_ind

                    mod_flag = True
                    self.modify_update()

            # Remove segment from cell track

            if flag == 3:

                self.track_btn3.state = 'normal'

                if sum(mask) and min(d) < 50:
                    self.features['tracking'][self.features['tracking'][:, 0] == sel[0], 11] = 1

                    ind = np.where(mask)
                    self.tracks = np.delete(self.tracks, ind[0][0], 0)

                    mod_flag = True
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

                    if swap_ind == self.track_ind[0]:
                        return

                    swap_mask = self.tracks[:, 4] == swap_ind

                    if not np.count_nonzero(swap_mask):
                        return

                    swap_track = self.tracks[swap_mask, :]

                    self.tracks = self.tracks[np.logical_not(np.logical_or(sel_mask, swap_mask)), :]

                    # perform swap

                    sel1 = sel_track[sel_track[:, 5] < self.current_frame, :]
                    sel2 = sel_track[sel_track[:, 5] >= self.current_frame, :]

                    swap1 = swap_track[swap_track[:, 5] < self.current_frame, :]
                    swap2 = swap_track[swap_track[:, 5] >= self.current_frame, :]

                    swapped_1 = np.zeros(1)
                    swapped_2 = np.zeros(1)

                    if np.count_nonzero(sel1) and np.count_nonzero(swap2):
                        swapped_1 = np.vstack((sel1, swap2))
                    else:
                        if np.count_nonzero(sel1):
                            swapped_1 = sel1
                        if np.count_nonzero(swap2):
                            swapped_1 = swap2

                    if np.count_nonzero(swap1) and np.count_nonzero(sel2):
                        swapped_2 = np.vstack((swap1, sel2))
                    else:
                        if np.count_nonzero(swap1):
                            swapped_2 = swap1
                        if np.count_nonzero(sel2):
                            swapped_2 = sel2

                    if np.count_nonzero(swapped_1):
                        if np.count_nonzero(sel1):
                            swapped_1[:, 4] = sel1[0, 4]
                        if np.count_nonzero(sel2):
                            swapped_1[:, 4] = sel2[0, 4]

                    if np.count_nonzero(swapped_2):
                        if np.count_nonzero(swap1):
                            swapped_2[:, 4] = swap1[0, 4]
                        if np.count_nonzero(swap2):
                            swapped_2[:, 4] = swap2[0, 4]


                    # update labels

                    if np.count_nonzero(swapped_1):
                        self.tracks = np.vstack((self.tracks, swapped_1))
                        map_ind1 = self.features['tracking'][int(swapped_1[0, 0]), 11]
                        self.features['tracking'][swapped_1[:, 0].astype(int), 11] = map_ind1

                    if np.count_nonzero(swapped_2):
                        self.tracks = np.vstack((self.tracks, swapped_2))
                        map_ind2 = self.features['tracking'][int(swapped_2[0, 0]), 11]
                        self.features['tracking'][swapped_2[:, 0].astype(int), 11] = map_ind2

                    mod_flag = True
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
                    self.features['tracking'][self.features['tracking'][:, 0] == sel[0], 11] = r

                    self.modify_update()

                    stored_temp = self.parent.fov['tracks_stored'][...]
                    stored_temp = np.append(stored_temp, 0)

                    del self.parent.fov['tracks_stored']
                    self.parent.fov.create_dataset("tracks_stored", data=stored_temp)
                    self.store_layout.add_widget(CellMark(size_hint=(.43, .43), pos_hint={'x': .12, 'y': .46}))

                    mod_flag = True

        if mod_flag:
            del self.parent.fov['tracks']
            self.parent.fov.create_dataset("tracks", data=self.tracks)

    def keyboard_closed(self):
        self.keyboard.unbind(on_key_down=self.key_print)
        self.keyboard = None

    def key_print(self, keyboard, keycode, text, modifiers):

        key = keycode[1]

        if key == 'a':
            self.frame_select([], self.current_frame - 1)
            self.canvas.ask_update()

        if key == 'd':
            self.frame_select([], self.current_frame + 1)
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

            self.tracking_window.keyboard_press(self.track_btn6.state, 5)

        if key == 'w':

            if self.track_btn5.state == 'normal':
                self.track_btn5.state = 'down'

            else:
                self.track_btn5.state = 'normal'

            self.jump_window.k_jump(self.track_btn5.state, flag=0)

        if key == 'i':

            if self.event_flag1.state == 'normal':
                self.event_flag1.state = 'down'

            else:
                self.event_flag1.state = 'normal'

            self.jump_window.k_jump(self.event_flag1.state, flag=1)

        if key == 'o':

            if self.event_flag2.state == 'normal':
                self.event_flag2.state = 'down'

            else:
                self.event_flag2.state = 'normal'

            self.jump_window.k_jump(self.event_flag2.state, flag=2)

        if key == 'p':

            if self.event_flag3.state == 'normal':
                self.event_flag3.state = 'down'

            else:
                self.event_flag3.state = 'normal'

            self.jump_window.k_jump(self.event_flag3.state, flag=3)

    def store_track(self, instance):

        if self.parent.fov['tracks_stored'][int(self.track_ind)] == 0:
            self.parent.fov['tracks_stored'][int(self.track_ind)] = 1
            self.store_ids = self.store_ids.union(set(self.tracks[self.tracks[:, 4] == self.track_ind, 0]))

        else:
            self.parent.fov['tracks_stored'][int(self.track_ind)] = 0
            self.store_ids = self.store_ids.difference(set(self.tracks[self.tracks[:, 4] == self.track_ind, 0]))

        self.modify_update()

    def save_csv(self, instance):

        trackcells.save_csv(self.features, self.tracks, self.parent.csv_file)
        #trackcells.save_iscb(self.features, self.tracks, self.parent.csv_file, self.labels, self.frames) # Need to update since changes

    def save_sel_csv(self, instance):

        trackcells.save_sel_csv(self.features, self.tracks, self.parent.fov['tracks_stored'], self.parent.sel_csv_file)

    def feat_change(self, flag, instance):

        if instance.text.isdigit():
            num = int(''.join([instance.text]))-1

            if 0 <= num < self.features['data'].shape[1]:

                self.show_feat[flag] = num

            self.modify_update()

            self.keyboard = Window.request_keyboard(self.keyboard_closed, self)
            self.keyboard.bind(on_key_down=self.key_print)

    def change_channel(self, val, instance):

        self.channel = int(val)
        im = np.asarray(Image.open(self.file_list[self.channel][self.current_frame]))
        im = im.astype(float)
        self.mov_disp.update_im(im)


    def remove(self):

        self.tr_layout.clear_widgets()
        self.store_layout.clear_widgets()

    def update_size(self, window, width, height):
        self.tr_layout.width = width
        self.tr_layout.height = height

        self.store_layout.width = width
        self.store_layout.height = height
