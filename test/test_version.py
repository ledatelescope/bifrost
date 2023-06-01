import unittest
import subprocess
import sys

class TestVersion(unittest.TestCase):
    def test_plain_version(self):
        subprocess.check_output([sys.executable, '-m', 'bifrost.version'])

    def test_version_config(self):
        subprocess.check_output([sys.executable, '-m', 'bifrost.version', '--config'])
