from kivy.uix.togglebutton import ToggleButton
from kivy.uix.togglebutton import Button
from kivy.uix.popup import Popup
from kivy.uix.label import Label


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

def error_msg(message):

    error_msg = Popup(title='Error message',
                      content=Label(text=message),
                      size_hint=(0.6, 0.3))
    error_msg.open()
