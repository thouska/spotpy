"""
This creates a simple gui for spotpy setups using the portable widgets
of matplotlib.widgets. This part of spotpy is NOT Python 2 compatible
and needs Python 3.5 minimum

:author: Philipp Kraft (https://philippkraft.github.io)
"""

import sys
import matplotlib

from matplotlib.widgets import Slider, Button
from matplotlib import pylab as plt

import time
from ..parameter import get_parameters_array, create_set


if sys.version_info < (3, 5):
    raise ImportError('spotpy.gui.mpl needs at least Python 3.5, you are running Python {}.{}.{}'
                      .format(*sys.version_info[:3]))


if matplotlib.__version__ < '2.1':
    raise ImportError('Your matplotlib package is too old. Required >=2.1, you have ' + matplotlib.__version__)


def as_scalar(val):
    """
    If val is iterable, this function returns the first entry
    else returns val. Handles strings as scalars
    :param val: A scalar or iterable
    :return:
    """
    # Check string
    if val == str(val):
        return val

    try:  # Check iterable
        it = iter(val)
        # Get first iterable
        return next(it)
    except TypeError:
        # Fallback, val is scalar
        return val


class Widget:
    """
    A class for simple widget building in matplotlib

    Takes events as keywords on creation

    Usage:
    >>>from matplotlib.widgets import Button
    >>>def when_click(_): print('Click')
    >>>w = Widget([0,0,0.1,0.1], Button, 'click me', on_clicked=when_click)
    """

    def __init__(self, rect, wtype, *args, **kwargs):
        """
        Creates a matplotlib.widgets widget
        :param rect: The rectangle of the position [left, bottom, width, height] in relative figure coordinates
        :param wtype: A type from matplotlib.widgets, eg. Button, Slider, TextBox, RadioButtons
        :param args: Positional arguments passed to the widget
        :param kwargs: Keyword arguments passed to the widget and events used for the widget
                       eg. if wtype is Slider, on_changed=f can be used as keyword argument

        """
        self.ax = plt.axes(rect)
        events = {}
        for k in list(kwargs.keys()):
            if k.startswith('on_'):
                events[k] = kwargs.pop(k)
        self.object = wtype(self.ax, *args, **kwargs)
        for k in events:
            if hasattr(self.object, k):
                getattr(self.object, k)(events[k])


class ValueChanger:
    """
    A closure class to change values by key with the sliders.
    Used as eventhandler for the sliders

    Usage:

    >>>d = {}
    >>>from matplotlib.widgets import Slider
    >>>w = Widget([0,0,0.1,0.1], Slider, 'slider', 0, 100, on_changed=ValueChanger('a', d))
    """
    def __init__(self, key, stor):
        self.stor = stor
        self.key = key

    def __call__(self, val):
        self.stor[self.key] = val


class GUI:
    """
    A simple graphic user interface for a setup, for manual calibration.

    Until now, the graph is not super nice, and gets confusing with multiple timeseries
    Uses the simulation, evaluation and objectivefunction methods of the spotpy setup.

    Fails for multiobjective setups, the return value of objectivefunction must be scalar

    Usage:

    >>>from spotpy.gui.mpl import GUI
    >>>from spotpy.examples.spot_setup_rosenbrock import spot_setup
    >>>gui = GUI(spot_setup())
    >>>gui.show()

    """

    def __init__(self, setup):
        """
        Creates the GUI

        :param setup: A spotpy setup
        """
        self.fig = plt.figure(type(setup).__name__)
        self.ax = plt.axes([0.05, 0.1, 0.65, 0.85])
        self.button_run = Widget([0.75, 0.01, 0.1, 0.03], Button, 'Simulate', on_clicked=self.run)
        self.button_clear = Widget([0.87, 0.01, 0.1, 0.03], Button, 'Clear plot', on_clicked=self.clear)
        self.parameter_values = {}
        self.setup = setup
        self.sliders = self._make_widgets()
        self.lines = []
        self.clear()

    def close(self):
        plt.close(self.fig)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    @staticmethod
    def show():
        """
        Calls matplotlib.pylab.show to show the GUI.
        """
        plt.show()

    def _make_widgets(self):
        """
        Creates the sliders
        :return:
        """
        if hasattr(self, 'sliders'):
            for s in self.sliders:
                s.ax.remove()

        sliders = []
        step = max(0.005, min(0.05, 0.8/len(self.parameter_array)))
        for i, row in enumerate(self.parameter_array):
            rect = [0.75, 0.9 - step * i, 0.2, step - 0.005]
            s = Widget(rect, Slider, row['name'], row['minbound'], row['maxbound'],
                       valinit=row['optguess'], on_changed=ValueChanger(row['name'], self.parameter_values))
            sliders.append(s)
        plt.draw()
        return sliders

    @property
    def parameter_array(self):
        return get_parameters_array(self.setup)

    def clear(self, _=None):
        """
        Clears the graph and plots the evalution
        """
        obs = self.setup.evaluation()
        self.ax.clear()
        self.lines = list(self.ax.plot(obs, 'k:', label='Observation', zorder=2))
        self.ax.legend()

    def run(self, _=None):
        """
        Runs the model and plots the result
        """
        self.ax.set_title('Calculating...')
        plt.draw()
        time.sleep(0.001)

        parset = create_set(self.setup, **self.parameter_values)
        sim = self.setup.simulation(parset)
        objf = as_scalar(self.setup.objectivefunction(sim, self.setup.evaluation()))
        label = ('{:0.4g}=M('.format(objf)
                 + ', '.join('{f}={v:0.4g}'.format(f=f, v=v) for f, v in zip(parset.name, parset))
                 + ')')
        self.lines.extend(self.ax.plot(sim, '-', label=label))
        self.ax.legend()
        self.ax.set_title(type(self.setup).__name__)
        plt.draw()
