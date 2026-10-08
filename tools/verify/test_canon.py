"""Regression tests for the normalizer's proof context.

    python3 -m unittest discover -s tools/verify -p test_canon.py
"""

import unittest
from z3 import BitVecs, Extract
import canon


class ContextTests(unittest.TestCase):
    def tearDown(self):
        canon.set_context([], set())

    def test_iteration_counter_is_a_parameter(self):
        counter, iteration = BitVecs("counter len_k", 64)
        canon.set_context([], {"len_k"})
        whole = Extract(31, 0, counter + 16 * iteration)
        halves = Extract(31, 0, counter) + 16 * Extract(31, 0, iteration)
        self.assertEqual(canon.canon(whole), canon.canon(halves))
        self.assertNotEqual(canon.canon(whole), canon.canon(halves + 1))

    def test_context_resets_symbol_classification(self):
        _, iteration = BitVecs("counter len_k", 64)
        self.assertTrue(canon.has_data(iteration))
        canon.set_context([], {"len_k"})
        self.assertFalse(canon.has_data(iteration))
        canon.set_context([], set())
        self.assertTrue(canon.has_data(iteration))


if __name__ == "__main__":
    unittest.main()
