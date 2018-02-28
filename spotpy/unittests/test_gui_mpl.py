
import unittest
import matplotlib
matplotlib.use('Agg')

import sys

if sys.version_info >= (3, 5) or matplotlib.__version__ >= '2.1':

    sys.path.append(".")


    try:
        import spotpy
    except ImportError:
        import spotpy

    from spotpy.gui.mpl import GUI
    from test_setup_parameters import SpotSetupBase


    class TestGuiMpl(unittest.TestCase):

        def test_setup(self):
            for cls in SpotSetupBase.get_derived():
                setup = cls()
                with GUI(setup) as gui:
                    self.assertTrue(hasattr(gui, 'setup'))

        def test_sliders(self):
            for cls in SpotSetupBase.get_derived():
                setup = cls()
                with GUI(setup) as gui:
                    self.assertEqual(len(gui.sliders), 4)

        def test_clear(self):
            for cls in SpotSetupBase.get_derived():
                setup = cls()
                with GUI(setup) as gui:
                    gui.clear()
                    self.assertEqual(len(gui.lines), 1)

        def test_run(self):
            for cls in SpotSetupBase.get_derived():
                setup = cls()
                with GUI(setup) as gui:
                    gui.clear()
                    self.assertEqual(len(gui.lines), 1)

if __name__ == '__main__':
    unittest.main()