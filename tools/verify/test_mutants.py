"""Mutation rejection is a proof result; checker crashes fail the run."""

import unittest

from aarch64 import Unproved
from mutants import run


class MutationResults(unittest.TestCase):
    def test_accepted_mutant_fails(self):
        _, failed = run(("control", None, lambda: None))
        self.assertTrue(failed)

    def test_rejected_mutant_passes(self):
        def reject():
            raise Unproved("control rejection")
        line, failed = run(("control", None, reject))
        self.assertFalse(failed)
        self.assertIn("control rejection", line)

    def test_main_fails_when_a_chunk_root_mutant_is_accepted(self):
        import io
        import mutants
        import workers
        from unittest.mock import patch
        jobs = {name: (lambda: []) for name in ("jobs", "last_mutant_jobs", "x86_mutant_jobs", "x86_memory_mutant_jobs", "wide_mutant_jobs")}
        jobs.update({name: (lambda _: None) for name in ("last_positive", "x86_positive", "x86_memory_positive")})
        jobs["chunk_root_mutant_jobs"] = lambda: [("accepted fault", None, lambda: None)]
        with patch.multiple(mutants, **jobs), patch.object(workers, "each", lambda fn, items, *_: (fn(item) for item in items)), patch("sys.stdout", new_callable=io.StringIO):
            with self.assertRaises(SystemExit) as stopped:
                mutants.main()
            self.assertEqual(stopped.exception.code, 1)

    def test_checker_error_propagates(self):
        def broken():
            raise KeyError("checker defect")
        with self.assertRaises(KeyError):
            run(("control", None, broken))


if __name__ == "__main__":
    unittest.main()
