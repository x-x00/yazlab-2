import json
import numpy as np
from typing import List, Dict, Any, Optional
import Levenshtein

# core explainer
class AutomataExplainer:

    def __init__(self, automata_model):
        self.model = automata_model

    # full sequence explanation

    def explain_sequence(self, signal_1d: np.ndarray,
                         y_true: Optional[np.ndarray] = None
                         ) -> List[Dict[str, Any]]:
        from models.automata_model import signal_to_sax, extract_patterns

        sax = signal_to_sax(signal_1d,
                             self.model.window_size,
                             self.model.alphabet_size)
        patterns = extract_patterns(sax, self.model.pattern_len)

        cumulative_prob = 1.0
        reports = []

        for t in range(len(patterns) - 1):
            raw_src = patterns[t]
            raw_dst = patterns[t + 1]

            src, src_status, src_dist = self.model.resolve_pattern(raw_src)
            dst, dst_status, _        = self.model.resolve_pattern(raw_dst)

            step_prob = self.model.transitions.get(src, {}).get(dst, 1e-6)

            cumulative_prob *= step_prob

            decision = "anomaly" if step_prob < self.model.anomaly_threshold else "normal"

            report: Dict[str, Any] = {
                "time_step":        t,
                "state":            src,
                "pattern":          raw_src,
                "status":           src_status,
                "mapped_to":        src,
                "edit_distance":    src_dist,
                "transitions": {
                    f"{src} -> {dst}": round(step_prob, 6)
                },
                "path_probability": round(cumulative_prob, 6),
                "probability":      round(cumulative_prob, 6),
                "decision":         decision,
                "confidence_score": round(step_prob, 6),
            }

            if y_true is not None and t < len(y_true):
                report["ground_truth"] = int(y_true[t])

            reports.append(report)

        return reports

    # path probability for a full pattern sequence
    def path_probability(self, patterns: List[str]) -> float:
        if len(patterns) < 2:
            return 1.0
        prob = 1.0
        for i in range(len(patterns) - 1):
            src, _, _ = self.model.resolve_pattern(patterns[i])
            dst, _, _ = self.model.resolve_pattern(patterns[i + 1])
            p = self.model.transitions.get(src, {}).get(dst, 1e-6)
            prob *= p
        return prob

    # confidence score
    def confidence_score(self, signal_1d: np.ndarray) -> float:
        proba, _ = self.model.predict_proba(signal_1d)
        if len(proba) == 0:
            return 0.0
        log_proba = np.log(np.maximum(proba, 1e-12))
        return float(np.exp(log_proba.mean()))

    # similarity-based explanation (optional advanced)
    def similarity_report(self, pattern: str) -> Dict[str, Any]:
        distances = {
            voc: Levenshtein.distance(pattern, voc)
            for voc in self.model.vocabulary
        }
        sorted_d = sorted(distances.items(), key=lambda x: x[1])
        return {
            "query_pattern": pattern,
            "nearest": sorted_d[:5],
            "in_vocabulary": pattern in self.model.vocabulary,
        }

    # counterfactual analysis (optional advanced)
    def counterfactual(self, pattern: str, target_decision: str = "normal"
                       ) -> Dict[str, Any]:
        target_anomaly = target_decision == "anomaly"
        candidates = []
        for voc_pat in self.model.vocabulary:
            out_probs = list(self.model.transitions.get(voc_pat, {}).values())
            if not out_probs:
                continue
            avg_p = np.mean(out_probs)
            is_anomaly = avg_p < self.model.anomaly_threshold
            if is_anomaly == target_anomaly:
                dist = Levenshtein.distance(pattern, voc_pat)
                candidates.append((voc_pat, dist, avg_p))
        candidates.sort(key=lambda x: x[1])
        return {
            "original_pattern": pattern,
            "target_decision":  target_decision,
            "counterfactual_candidates": [
                {"pattern": p, "edit_distance": d, "avg_out_prob": round(a, 4)}
                for p, d, a in candidates[:3]
            ],
        }

    # formatted text report
    @staticmethod
    def format_step(report: Dict[str, Any]) -> str:
        lines = [
            "[SYSTEM DECISION]",
            f"Time Step:        t = {report['time_step']}",
            f"Previous State:   \"{report['state']}\"",
            f"Incoming Pattern: \"{report['pattern']}\"",
            f"Status:           {report['status']}",
        ]
        if report["status"] == "unseen":
            lines.append(f"Nearest Pattern:  \"{report['mapped_to']}\" "
                         f"(distance = {report['edit_distance']})")
        lines.append("Transitions:")
        for trans, prob in report["transitions"].items():
            lines.append(f"  {trans} : {prob}")
        lines += [
            f"Path Probability: {report['path_probability']}",
            f"Decision:         {report['decision'].upper()}",
            f"Confidence Score: {report['confidence_score']} "
            f"({'Low' if report['confidence_score'] < 0.1 else 'Medium' if report['confidence_score'] < 0.5 else 'High'})",
        ]
        return "\n".join(lines)

    # JSON export
    @staticmethod
    def to_json(report: Dict[str, Any]) -> str:
        return json.dumps(report, indent=2)
