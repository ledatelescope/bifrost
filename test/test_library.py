import unittest

from bifrost.libbifrost_generated import bfTestSuite

class TestLibrary(unittest.TestCase):
    def test_library(self):
        self.assertEqual(bfTestSuite(), 0)
