import sys
import os
import argparse
import json
import unittest

sys.path.append(os.path.abspath("../rag/"))

parser = argparse.ArgumentParser()
parser.add_argument('integers', metavar='N', type=int, nargs=1,
                    help='an integer for the accumulator')
print(sys.argv)
args = parser.parse_args()

# Load the JSON data from your file
with open('test' + str(args.integers[0]) + '.json') as json_file:
    test_args = json.load(json_file)['testParameters']

class TestStringMethods(unittest.TestCase):

    def test_upper(self):
        self.assertEqual('foo'.upper(), 'FOO')

    def test_isupper(self):
        self.assertTrue('FOO'.isupper())
        self.assertFalse('Foo'.isupper())

    def test_split(self):
        s = 'hello world'
        self.assertEqual(s.split(), ['hello', 'world'])
        # check that s.split fails when the separator is not a string
        with self.assertRaises(TypeError):
            s.split(2)

if __name__ == '__main__':
    unittest.main()