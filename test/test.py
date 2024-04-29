import unittest
import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '../rag/rag.py')
from rag import extract_data

class TestExtract(unittest.TestCase):

    def test_extract(self):
        doc = extract_data()
        self.assertEqual('FOO', 'FOO')

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