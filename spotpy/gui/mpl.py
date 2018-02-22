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


class GUI:
    """
    The graphic user interface for a setup, for manual calibration.
    Does not work with setups using a parameters function
    """

    def __init__(self, setup):
        self.fig = plt.figure()
        self.ax = plt.axes([0.3, 0.1, 0.6, 0.8])
        self.button = Widget([0.05, 0.01, 0.2, 0.04], Button, 'Simulate', on_clicked=self.run)
        self.parameter_values = {}
        self.setup = setup
        self.sliders=[]
        self.make_widgets()
        self.run()

    def show(self):
        print('GUI')
        plt.show()

    def make_widgets(self):
        for s in self.sliders:
            s.ax.remove()

        self.sliders = []
        for i, row in enumerate(self.parameter_array):
            f = lambda val: self.change_parameter(p.name, val)
            rect = [0.05, 0.9 - 0.05 * i, 0.2, 0.04]
            s = Widget(rect, Slider, row['name'], row['minbound'], row['maxbound'],
                       valinit=row['optguess'], on_changed=f)
            self.sliders.append(s)
        plt.draw()

    @property
    def parameter_array(self):
        return get_parameters_array(self.setup)

    def change_parameter(self, key, value):
        self.parameter_values[key] = value

    def run(self):
        parset = create_set(self.setup, **self.parameter_values)
        sim = self.setup.simulation(parset)
        obs = self.setup.evaluation()
        self.ax.plot(sim, '-', label='Simulation')
        self.ax.plot(obs, '--', label='Observation')
        plt.draw()

