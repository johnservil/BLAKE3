"""Regression tests for the proofs' helpers: the normalizer's proof
context (canon.py) and the induction's generalization (induction.py).

    python3 -m unittest discover -s tools/verify -p test_helpers.py
"""

import unittest
from z3 import BitVec, BitVecs, Concat, Extract
import canon
import induction


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

    def test_length_parts_of_a_sum_combine(self):
        # A word plus a group number the kernel computes from an address,
        # against the same word plus K + 1: one normal form; K + 2 differs.
        from z3 import BitVec, LShR, ULE
        word, k = BitVec("pair0", 32), BitVec("len_k", 64)
        canon.set_context([ULE(k, 1 << 39)], {"len_k"})
        computed = word + Extract(31, 0, LShR(1024 * k + 1024 + 5, 10))
        self.assertEqual(canon.canon(computed), canon.canon(word + Extract(31, 0, k) + 1))
        self.assertNotEqual(canon.canon(computed), canon.canon(word + Extract(31, 0, k) + 2))

    def test_context_resets_symbol_classification(self):
        _, iteration = BitVecs("counter len_k", 64)
        self.assertTrue(canon.has_data(iteration))
        canon.set_context([], {"len_k"})
        self.assertFalse(canon.has_data(iteration))
        canon.set_context([], set())
        self.assertTrue(canon.has_data(iteration))


class GeneralizationTests(unittest.TestCase):
    def test_constant_difference_through_byte_shuffles(self):
        # A word loaded byte by byte: its bytes' Concat hides the difference
        # from Z3's simplifier.
        x = BitVec("pair0", 32)
        def loaded(t):
            return Concat(Extract(31, 24, t), Extract(23, 16, t), Extract(15, 8, t), Extract(7, 0, t))
        self.assertEqual(induction.constant_difference(loaded(x), loaded(x + 1)), 1)
        self.assertIsNone(induction.constant_difference(x, x * x))


if __name__ == "__main__":
    unittest.main()
