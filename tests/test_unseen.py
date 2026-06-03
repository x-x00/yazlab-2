import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import numpy as np
import Levenshtein
from models.automata_model import ProbabilisticAutomata, signal_to_sax, extract_patterns


class TestLevenshtein(unittest.TestCase):

    def test_identical_strings(self):
        self.assertEqual(Levenshtein.distance("abc", "abc"), 0)

    def test_single_substitution(self):
        self.assertEqual(Levenshtein.distance("abc", "adc"), 1)

    def test_insertion(self):
        self.assertEqual(Levenshtein.distance("ab", "abc"), 1)

    def test_deletion(self):
        self.assertEqual(Levenshtein.distance("abc", "ab"), 1)

    def test_complete_replacement(self):
        d = Levenshtein.distance("aaa", "bbb")
        self.assertEqual(d, 3)

    def test_empty_strings(self):
        self.assertEqual(Levenshtein.distance("", ""), 0)
        self.assertEqual(Levenshtein.distance("", "abc"), 3)


class TestUnseenPatternResolution(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        signal = np.sin(np.linspace(0, 4 * np.pi, 200))
        self.automata = ProbabilisticAutomata(window_size=4, alphabet_size=3,
                                              pattern_len=2)
        self.automata.fit(signal)

    def test_vocabulary_not_empty(self):
        self.assertGreater(len(self.automata.vocabulary), 0)

    def test_seen_pattern(self):
        if self.automata.vocabulary:
            pat = next(iter(self.automata.vocabulary))
            _, status, dist = self.automata.resolve_pattern(pat)
            self.assertEqual(status, "seen")
            self.assertIsNone(dist)

    def test_unseen_pattern(self):
        unseen = "zzzzzzzzz"
        _, status, dist = self.automata.resolve_pattern(unseen)
        self.assertEqual(status, "unseen")
        self.assertIsNotNone(dist)
        self.assertGreater(dist, 0)

    def test_nearest_pattern_is_valid(self):
        unseen = "zzz"
        nearest, dist = self.automata.nearest_pattern(unseen)
        self.assertIn(nearest, self.automata.vocabulary)

    def test_nearest_pattern_minimises_distance(self):
        unseen = "zzz"
        nearest, dist = self.automata.nearest_pattern(unseen)

        for voc_pat in self.automata.vocabulary:
            d = Levenshtein.distance(unseen, voc_pat)
            self.assertGreaterEqual(d, dist)

    def test_exact_match_preferred(self):
        if self.automata.vocabulary:
            pat = next(iter(self.automata.vocabulary))
            nearest, dist = self.automata.nearest_pattern(pat)
            self.assertEqual(dist, 0)
            self.assertEqual(nearest, pat)


class TestSAXPipeline(unittest.TestCase):

    def test_paa_output_length(self):
        from models.automata_model import paa
        signal = np.random.randn(100)
        result = paa(signal, window_size=4)
        self.assertEqual(len(result), 25)

    def test_sax_symbols_in_alphabet(self):
        from models.automata_model import sax_encode
        import string
        paa_vals = np.array([-1.0, -0.2, 0.2, 1.0])
        symbols = sax_encode(paa_vals, alphabet_size=3)
        for s in symbols:
            self.assertIn(s, string.ascii_lowercase)

    def test_pattern_extraction(self):
        sax = ["a", "b", "c", "d", "e"]
        patterns = extract_patterns(sax, pattern_len=2)
        self.assertEqual(patterns, ["ab", "bc", "cd", "de"])

    def test_signal_to_sax_length(self):
        signal = np.random.randn(200)
        sax = signal_to_sax(signal, window_size=4, alphabet_size=3)
        self.assertEqual(len(sax), 50)  # 200 // 4


class TestProbabilisticAutomata(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        self.signal = np.concatenate([
            np.sin(np.linspace(0, 6 * np.pi, 300)),
            np.random.randn(50) * 2
        ])
        self.automata = ProbabilisticAutomata(window_size=4, alphabet_size=3,
                                              pattern_len=2)
        self.automata.fit(self.signal[:300])

    def test_transitions_sum_to_one(self):
        for src, dst_probs in self.automata.transitions.items():
            total = sum(dst_probs.values())
            self.assertLessEqual(total, 1.0 + 1e-9,
                                 msg=f"'{src}' transitions are exceeding 1: {total}")
            self.assertGreater(total, 0.8,
                               msg=f"'{src}' transitions are too low: {total}")

    def test_predict_proba_returns_array(self):
        proba, explanations = self.automata.predict_proba(self.signal[300:])
        self.assertIsInstance(proba, np.ndarray)
        self.assertGreater(len(proba), 0)

    def test_state_count_positive(self):
        self.assertGreater(self.automata.state_count(), 0)

    def test_transition_density_between_0_and_1(self):
        density = self.automata.transition_density()
        self.assertGreaterEqual(density, 0.0)
        self.assertLessEqual(density, 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
