from kivy.uix.togglebutton import ToggleButton
from kivy.uix.togglebutton import Button
from kivy.uix.popup import Popup
from kivy.uix.label import Label
from kivy.uix.slider import Slider
from kivy.uix.widget import Widget
from kivy.uix.floatlayout import FloatLayout

from functools import partial


def add_tbtn(layout, text, func, group=None):

    btn = ToggleButton(text=text, group=group, halign='center', valign='middle')
    btn.bind(on_press=func)
    btn.bind(size=btn.setter('text_size'))
    layout.add_widget(btn)

    return


def add_btn(layout, text, func, group=None):
    btn = Button(text=text, group=group, halign='center', valign='middle')
    btn.bind(on_press=func)
    btn.bind(size=btn.setter('text_size'))
    layout.add_widget(btn)

    return


def notify_msg(message):

    msg = Popup(title='Message',
                      content=Label(text=message),
                      size_hint=(0.6, 0.3))
    msg.open()


def ntlabel(text, style, size_hint=None, pos_hint=None):

    style_on = ''
    style_off = ''

    if style == 1:
        style_on = '[b][color=000000]'
        style_off = '[/b][/color]'

    if style == 2:
        style_on = '[color=000000]'
        style_off = '[/color]'

    if size_hint is None:
        return Label(text=style_on + text + style_off, markup=True)
    else:
        return Label(text=style_on + text + style_off, markup=True,
              size_hint=size_hint, pos_hint=pos_hint)

def ntchange(label, style, text):

    style_on = ''
    style_off = ''

    if style == 1:
        style_on = '[b][color=000000]'
        style_off = '[/b][/color]'

    if style == 2:
        style_on = '[color=000000]'
        style_off = '[/color]'

    label.text = style_on + text + style_off

class FrameSlider(Widget):

    def __init__(self, frames, func, **kwargs):
        super().__init__(**kwargs)

        self.frame_slider = Slider(min=0, max=frames - 1, value=1, size_hint=(.7, .6), pos_hint={'x': .15, 'y': .4})
        self.frame_slider.bind(value=partial(self.change_frame, func))

        self.frame_text = ntlabel(text='Frame: ' + str(0), size_hint=(.6, .4), pos_hint={'x': .2, 'y': 0}, style=2)

        self.frame_minus = Button(text='<<',
                                  size_hint=(.15, .6), pos_hint={'x': .0, 'y': .4}, markup=True)
        self.frame_plus = Button(text='>>',
                                 size_hint=(.15, .6), pos_hint={'x': .85, 'y': .4}, markup=True)

        self.frame_minus.bind(on_press=self.frame_backward)
        self.frame_plus.bind(on_press=self.frame_forward)

        self.frames = frames
        self.slider_layout = FloatLayout(size=(500, 500), pos=self.pos)

        self.slider_layout.add_widget(self.frame_slider)
        self.slider_layout.add_widget(self.frame_text)
        self.slider_layout.add_widget(self.frame_minus)
        self.slider_layout.add_widget(self.frame_plus)

        with self.canvas:
            self.add_widget(self.slider_layout)

        self.bind(pos=self.update_size, size=self.update_size)  # Maintain image size on scaling of parent layout

    def frame_forward(self, instance):

        if self.frame_slider.value < self.frames - 1:
            self.frame_slider.value += 1

    def frame_backward(self, instance):

        if self.frame_slider.value > 0:
            self.frame_slider.value -= 1

    def change_frame(self, func, instance, val):

        func(int(val))
        ntchange(label=self.frame_text, text='Frame: ' + str(int(val)), style=2)

    def update_size(self, *args):

        self.slider_layout.pos = self.pos
        self.slider_layout.size = self.size

