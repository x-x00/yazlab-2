import numpy as np
import json
from collections import defaultdict
from typing import List, Dict, Tuple, Optional
import Levenshtein
from config import CFG


# paa
def paa(signal: np.ndarray, window_size: int) -> np.ndarray:
    # piecewise aggregate approximation. divides the signal into equal-length frames and returns the mean of each. truncates the tail if the signal length is not divisible by window_size
    n = len(signal)
    n_frames = n // window_size
    truncated = signal[: n_frames * window_size]
    return truncated.reshape(n_frames, window_size).mean(axis=1)


# sax

# breakpoints from the standard normal distribution for sax
_BREAKPOINTS = {
    2: [-0.0],
    3: [-0.4307, 0.4307],
    4: [-0.6745, 0.0000, 0.6745],
    5: [-0.8416, -0.2533, 0.2533, 0.8416],
    6: [-0.9674, -0.4307, 0.0000, 0.4307, 0.9674],
    7: [-1.0676, -0.5659, -0.1800, 0.1800, 0.5659, 1.0676],
    8: [-1.1503, -0.6745, -0.3186, 0.0000, 0.3186, 0.6745, 1.1503],
}

_ALPHABET = "abcdefghijklmnopqrstuvwxyz"


def sax_encode(paa_values: np.ndarray, alphabet_size: int) -> List[str]:
    # convert paa values to sax symbols using standard breakpoints. returns a list of single-character symbols
    if alphabet_size not in _BREAKPOINTS:
        raise ValueError(f"Unsupported alphabet_size={alphabet_size}. "
                         f"Choose from {list(_BREAKPOINTS)}")
    bps = _BREAKPOINTS[alphabet_size]
    symbols = []
    for v in paa_values:
        idx = np.searchsorted(bps, v, side="right")
        symbols.append(_ALPHABET[idx])
    return symbols


def signal_to_sax(signal: np.ndarray, window_size: int,
                  alphabet_size: int) -> List[str]:
    # pipeline: signal -> paa -> sax symbols
    paa_vals = paa(signal, window_size)
    return sax_encode(paa_vals, alphabet_size)


# pattern extraction (sliding window over SAX sequence)
def extract_patterns(sax_seq: List[str], pattern_len: int) -> List[str]:
    return ["".join(sax_seq[i: i + pattern_len])
            for i in range(len(sax_seq) - pattern_len + 1)]


# probabilistic automata
class ProbabilisticAutomata:

    def __init__(self, window_size: int = None, alphabet_size: int = None,
                 pattern_len: int = 2):
        self.window_size   = window_size   or CFG.automata.window_size_fixed
        self.alphabet_size = alphabet_size or CFG.automata.alphabet_size_fixed
        self.pattern_len   = pattern_len

        self.vocabulary: set       = set()
        self.transitions: Dict[str, Dict[str, float]] = {}
        self._trans_counts: Dict[str, Dict[str, int]]  = defaultdict(lambda: defaultdict(int))
        self.anomaly_threshold: float = CFG.automata.anomaly_threshold

    # training
    def fit(self, signal_1d: np.ndarray, smoothing: float = 1e-4):
        sax = signal_to_sax(signal_1d, self.window_size, self.alphabet_size)
        patterns = extract_patterns(sax, self.pattern_len)

        self.vocabulary = set(patterns)
        self._trans_counts.clear()

        for i in range(len(patterns) - 1):
            src = patterns[i]
            dst = patterns[i + 1]
            self._trans_counts[src][dst] += 1

        vocab_size = len(self.vocabulary)
        self.transitions = {}
        for src, dst_counts in self._trans_counts.items():
            total = sum(dst_counts.values())

            self.transitions[src] = {
                dst: (cnt + smoothing) / (total + smoothing * vocab_size)
                for dst, cnt in dst_counts.items()
            }

        train_proba = []
        for i in range(len(patterns) - 1):
            src = patterns[i]; dst = patterns[i + 1]
            p = self.transitions.get(src, {}).get(dst, smoothing / (1 + smoothing * vocab_size))
            train_proba.append(p)
        if train_proba:
            self.anomaly_threshold = float(np.percentile(train_proba, 10))

        return self

    # unseen pattern matching
    def nearest_pattern(self, pattern: str) -> Tuple[str, int]:
        if not self.vocabulary:
            return pattern, 0
        best, best_dist = None, float("inf")
        for voc_pat in self.vocabulary:
            d = Levenshtein.distance(pattern, voc_pat)
            if d < best_dist:
                best_dist = d
                best = voc_pat
        return best, best_dist

    def resolve_pattern(self, pattern: str) -> Tuple[str, str, Optional[int]]:
        if pattern in self.vocabulary:
            return pattern, "seen", None
        nearest, dist = self.nearest_pattern(pattern)
        return nearest, "unseen", dist

    # prediction
    def predict_proba(self, signal_1d: np.ndarray
                      ) -> Tuple[np.ndarray, List[dict]]:
        sax = signal_to_sax(signal_1d, self.window_size, self.alphabet_size)
        patterns = extract_patterns(sax, self.pattern_len)

        scores       = []
        explanations = []

        for t in range(len(patterns) - 1):
            raw_src = patterns[t]
            raw_dst = patterns[t + 1]

            src, src_status, src_dist = self.resolve_pattern(raw_src)
            dst, dst_status, dst_dist = self.resolve_pattern(raw_dst)

            prob = self.transitions.get(src, {}).get(dst, 1e-6)

            is_anomaly = int(prob < self.anomaly_threshold)

            expl = {
                "time_step":   t,
                "state":       src,
                "pattern":     raw_src,
                "status":      src_status,
                "mapped_to":   src if src_status == "seen" else src,
                "next_state":  dst,
                "next_pattern": raw_dst,
                "edit_distance": src_dist,
                "probability": round(prob, 6),
                "decision":    "anomaly" if is_anomaly else "normal",
                "confidence":  round(prob, 6),
            }
            explanations.append(expl)
            scores.append(prob)

        return np.array(scores), explanations

    def predict(self, signal_1d: np.ndarray) -> np.ndarray:
        
        proba, _ = self.predict_proba(signal_1d)
        labels = (proba < self.anomaly_threshold).astype(int)
        
        n_paa_frames = len(signal_1d) // self.window_size
        n_patterns   = max(n_paa_frames - self.pattern_len, 0)
        
        return labels

    def predict_on_windows(self, signal_1d: np.ndarray,
                           y_true: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        proba, _ = self.predict_proba(signal_1d)
        n_pred = len(proba)
        
        indices = np.linspace(0, len(y_true) - 1, n_pred, dtype=int)
        y_aligned = y_true[indices]
        y_pred    = (proba < self.anomaly_threshold).astype(int)
        return y_pred, y_aligned

    # state statistics

    def state_count(self) -> int:
        return len(self.vocabulary)

    def transition_density(self) -> float:
        n = len(self.vocabulary)
        if n < 2:
            return 0.0
        actual = sum(len(v) for v in self.transitions.values())
        return actual / (n * n)

    def get_transition_matrix(self) -> Tuple[List[str], np.ndarray]:
        states = sorted(self.vocabulary)
        n = len(states)
        idx = {s: i for i, s in enumerate(states)}
        mat = np.zeros((n, n))
        for src, dsts in self.transitions.items():
            if src not in idx:
                continue
            for dst, p in dsts.items():
                if dst in idx:
                    mat[idx[src], idx[dst]] = p
        return states, mat

    # persistence

    def save(self, path: str):
        import pickle, os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        state = self.__dict__.copy()
        state["_trans_counts"] = {k: dict(v)
                                  for k, v in self._trans_counts.items()}
        with open(path, "wb") as f:
            pickle.dump(state, f)

    @staticmethod
    def load(path: str) -> "ProbabilisticAutomata":
        import pickle
        with open(path, "rb") as f:
            state = pickle.load(f)
        obj = ProbabilisticAutomata.__new__(ProbabilisticAutomata)
        obj.__dict__.update(state)
        return obj


def automata_path(dataset: str, seed: int, scenario: str,
                  fold: int = None, base_dir: str = None) -> str:
    import os
    from config import CFG
    base = base_dir or os.path.join(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))), "models_saved")
    fold_str = f"_fold{fold}" if fold is not None else ""
    fname = f"Automata_seed{seed}{fold_str}_{scenario}.pkl"
    return os.path.join(base, dataset, fname)