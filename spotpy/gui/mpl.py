"""
This creates a simple gui for spotpy setups using the portable widgets
of matplotlib.widgets
"""
import matplotlib

if matplotlib.__version__ < '2.1':
    raise ImportError('Your matplotlib package is too old. Required >=2.1, you have ' + matplotlib.__version__)

from matplotlib.widgets import Slider, Button
from matplotlib import pylab as plt

from ..parameter import get_parameters_array, create_set

class Widget:
    """
    A class for simple widget building in matplotlib

    Takes events as keywords
    """

    def __init__(self, rect, wtype, *args, **kwargs):
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
    A simple class to change values by key with the sliders
    """
    def __init__(self, key, dict):
        self.dict = dict
        self.key = key

    def __call__(self, val):
        self.dict[self.key] = val


class GUI:
    """
    The graphic user interface for a setup, for manual calibration.

    Usage:
    >>>from spotpy.gui.mpl import GUI
    >>>gui = GUI(spotsetup)
    >>>gui.show()
    """

    def __init__(self, setup):
        """
        Creates the widgets
        :param setup:
        """
        self.fig = plt.figure(str(setup))
        self.ax = plt.axes([0.05, 0.1, 0.65, 0.85])
        self.button_run = Widget([0.75, 0.01, 0.1, 0.03], Button, 'Simulate', on_clicked=self.run)
        self.button_clear = Widget([0.87, 0.01, 0.1, 0.03], Button, 'Clear plot', on_clicked=self.clear)
        self.parameter_values = {}
        self.setup = setup
        self.sliders = []
        self.make_widgets()
        self.clear()
        self.run()

    def show(self):
        plt.show()

    def make_widgets(self):
        for s in self.sliders:
            s.ax.remove()

        self.sliders = []
        for i, row in enumerate(self.parameter_array):
            rect = [0.75, 0.9 - 0.05 * i, 0.2, 0.04]
            s = Widget(rect, Slider, row['name'], row['minbound'], row['maxbound'],
                       valinit=row['optguess'], on_changed=ValueChanger(row['name'], self.parameter_values))
            self.sliders.append(s)
        plt.draw()

    @property
    def parameter_array(self):
        return get_parameters_array(self.setup)

    def clear(self, event=None):
        obs = self.setup.evaluation()
        self.ax.clear()
        self.ax.plot(obs, 'k:', label='Observation', zorder=2)

    def run(self, event=None):
        self.ax.set_title('Calculating...')
        plt.draw()

        parset = create_set(self.setup, **self.parameter_values)
        sim = self.setup.simulation(parset)
        objf = self.setup.objectivefunction(sim, self.setup.evaluation())
        label = ('{:0.4g}=M('.format(objf)
                 + ', '.join('{f}={v:0.4g}'.format(f=f, v=v) for f, v in zip(parset._fields, parset))
                 + ')')
        self.ax.plot(sim, '-', label=label)
        self.ax.legend()
        self.ax.set_title(str(self.setup))
        plt.draw()

