import os
import tempfile
import unittest
from itertools import product
from typing import Tuple

import numpy as np
from csnlp.util import io

from mpcrl.experience import ExperienceReplay

TMPFILENAME: str = ""


class TestExperienceReplay(unittest.TestCase):
    def test_init__initializes_memory_correctly(self):
        items = [object(), object(), object()]
        maxlen = 2
        mem = ExperienceReplay[object](items, maxlen=maxlen, seed=50)
        self.assertEqual(mem.maxlen, maxlen)
        self.assertNotIn(items[0], mem)
        self.assertIn(items[1], mem)
        self.assertIn(items[2], mem)
        self.assertIsInstance(mem.np_random, np.random.Generator)

    def test_experience__is_pickleable(self):
        mem = ExperienceReplay[Tuple[np.ndarray, float]]()
        for _ in range(10):
            mem.append((np.random.rand(10), np.random.rand()))
        self.assertTrue(io.is_pickleable(mem))

        global TMPFILENAME
        TMPFILENAME = next(tempfile._get_candidate_names())
        io.save(TMPFILENAME, ER=mem, check=69)
        mem_: ExperienceReplay = io.load(TMPFILENAME)["ER"]

        for (a, f), (a_, f_) in zip(mem, mem_):
            np.testing.assert_equal(a, a_)
            self.assertEqual(f, f_)

    def tearDown(self) -> None:
        try:
            os.remove(f"{TMPFILENAME}.pkl")
        finally:
            return super().tearDown()

    def test_sample__raises__with_no_maxlen_and_percentage_size(self):
        mem = ExperienceReplay[Tuple[np.ndarray, float]](maxlen=None)
        with self.assertRaises(TypeError):
            list(mem.sample(n=0.0))

    def test_sample__with_zero_samples__returns_no_samples(self):
        mem = ExperienceReplay[Tuple[np.ndarray, float]](maxlen=100)
        for n in (0, 0.0):
            self.assertListEqual(list(mem.sample(n)), [])

    def test_sample__returns_right_sample_size(self):
        N = 100
        Nsample = 10
        mem = ExperienceReplay[Tuple[np.ndarray, float]](maxlen=N)
        mem.extend(range(N))
        for n in (Nsample, Nsample / N):
            sample = list(mem.sample(n=n))
            self.assertEqual(len(sample), Nsample)
            for item in sample:
                self.assertIn(item, mem)

    def test_sample__last_n__includes_last_n_items(self):
        N = 100
        Nsample = 20
        Nlast = 10
        mem = ExperienceReplay[Tuple[np.ndarray, float]](maxlen=N)
        mem.extend(range(N))
        for n, last_n in product((Nsample, Nsample / N), (Nlast, Nlast / Nsample)):
            sample = list(mem.sample(n=n, last_n=last_n))
            self.assertEqual(len(sample), Nsample)
            for item in sample:
                self.assertIn(item, mem)
            for item in range(N - Nlast, N):
                self.assertIn(item, sample)


if __name__ == "__main__":
    unittest.main()