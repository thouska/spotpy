"""Tests for the unified CLI"""

import unittest

try:
    import spotpy
except ImportError:
    import sys
    sys.path.append("../../")
    sys.path.append("../../spotpy")
    sys.path.append(".")
    import spotpy
import spotpy.cli as cli

class TestDatabase(unittest.TestCase):

    def setUp(self):
        pass
