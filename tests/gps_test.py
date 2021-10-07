import sys
sys.path.insert(1, 'modules')
import gps

import unittest

class TestGPSMethods(unittest.TestCase):
    
    def test_calculate_heading_difference(self):    
        result = gps.galculate_path_distance((5,10),(20,40),(10,35))
        self.assertEqual(2826682.6387865897, result)\

if __name__ == '__main__':
    TestGPSMethods()
    print("Everythings passed!")