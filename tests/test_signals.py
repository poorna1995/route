import tempfile
import unittest
from pathlib import Path

from llm_routing.signals import (
    SIGNAL_TYPE_CROSS_MODEL,
    SIGNAL_TYPE_MODEL_RESPONSE,
    SignalRecord,
    load_signals,
    save_signals,
)


class TestSignals(unittest.TestCase):
    def test_roundtrip(self) -> None:
        rec = SignalRecord(
            "q1",
            signal_type=SIGNAL_TYPE_MODEL_RESPONSE,
            metrics={"entropy": 1.2, "margin": 0.8, "msp": 0.6},
            extra={"entropy_lo": 1.1, "entropy_hi": 0.7},
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "signals.jsonl"
            save_signals(path, [rec])
            loaded = load_signals(path)[0]
            self.assertEqual(loaded.query_id, "q1")
            self.assertEqual(loaded.signal_type, SIGNAL_TYPE_MODEL_RESPONSE)
            self.assertEqual(loaded.metrics["entropy"], 1.2)
            self.assertEqual(loaded.extra["entropy_lo"], 1.1)

    def test_legacy_flat_metrics(self) -> None:
        rec = SignalRecord.from_dict({"query_id": "q1", "entropy": 0.5, "msp": 0.9})
        self.assertEqual(rec.signal_type, SIGNAL_TYPE_MODEL_RESPONSE)
        self.assertEqual(rec.metrics["entropy"], 0.5)
        self.assertEqual(rec.metrics["msp"], 0.9)

    def test_explicit_signal_type(self) -> None:
        rec = SignalRecord.from_dict(
            {"query_id": "q1", "signal_type": SIGNAL_TYPE_CROSS_MODEL, "metrics": {"gap": 0.3}}
        )
        self.assertEqual(rec.signal_type, SIGNAL_TYPE_CROSS_MODEL)
        self.assertEqual(rec.metrics["gap"], 0.3)

    def test_unknown_keys_to_extra(self) -> None:
        rec = SignalRecord.from_dict(
            {"query_id": "q1", "signal_type": SIGNAL_TYPE_MODEL_RESPONSE, "disagreement": 1}
        )
        self.assertEqual(rec.extra["disagreement"], 1)

    def test_raw_roundtrip(self) -> None:
        rec = SignalRecord(
            "q1",
            signal_type=SIGNAL_TYPE_MODEL_RESPONSE,
            metrics={"entropy": 1.0},
            raw={"query": "What is 2+2?", "answer": "B"},
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "psi.jsonl"
            save_signals(path, [rec])
            loaded = load_signals(path)[0]
            self.assertEqual(loaded.raw["query"], "What is 2+2?")
            self.assertEqual(loaded.raw["answer"], "B")

    def test_legacy_prediction_keys(self) -> None:
        rec = SignalRecord.from_dict(
            {
                "query_id": "q1",
                "signal_type": SIGNAL_TYPE_MODEL_RESPONSE,
                "metrics": {"msp": 0.9},
                "prediction": {"letter": "B", "probability": 0.9},
            }
        )
        self.assertEqual(rec.prediction["parsed_answer"], "B")
        self.assertEqual(rec.prediction["confidence"], 0.9)

    def test_prediction_disagreement_bool(self) -> None:
        rec = SignalRecord.from_dict(
            {
                "query_id": "q1",
                "signal_type": SIGNAL_TYPE_CROSS_MODEL,
                "metrics": {"prediction_disagreement": True, "delta_entropy": 0.1},
            }
        )
        self.assertIs(rec.metrics["prediction_disagreement"], True)
        self.assertEqual(rec.metrics["delta_entropy"], 0.1)

        legacy = SignalRecord.from_dict(
            {
                "query_id": "q1",
                "signal_type": SIGNAL_TYPE_CROSS_MODEL,
                "metrics": {"prediction_disagreement": 1.0},
            }
        )
        self.assertIs(legacy.metrics["prediction_disagreement"], True)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "chi.jsonl"
            save_signals(path, [rec])
            raw = path.read_text(encoding="utf-8")
            self.assertIn('"prediction_disagreement": true', raw)
            self.assertNotIn('"prediction_disagreement": 1', raw)
        rec = SignalRecord.from_dict(
            {
                "query_id": "q1",
                "signal_type": SIGNAL_TYPE_MODEL_RESPONSE,
                "correct": 1,
                "parsed_answer": 2,
            }
        )
        self.assertEqual(rec.metrics, {})
        self.assertNotIn("correct", rec.extra)
        self.assertNotIn("parsed_answer", rec.extra)


if __name__ == "__main__":
    unittest.main()
