import tempfile
import unittest
from pathlib import Path

from llm_routing.signals import SignalRecord, load_signals, save_signals


class TestSignals(unittest.TestCase):
    def test_roundtrip(self) -> None:
        rec = SignalRecord(
            "q1", query_length=42, entropy=1.2, margin=0.8, msp=0.6, token_entropy=0.9,
            extra={"entropy_lo": 1.1, "entropy_hi": 0.7},
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "signals.jsonl"
            save_signals(path, [rec])
            loaded = load_signals(path)[0]
            self.assertEqual(loaded.query_id, "q1")
            self.assertEqual(loaded.entropy, 1.2)
            self.assertEqual(loaded.extra["entropy_lo"], 1.1)

    def test_unknown_keys_to_extra(self) -> None:
        rec = SignalRecord.from_dict({"query_id": "q1", "disagreement": 1})
        self.assertEqual(rec.extra["disagreement"], 1)


if __name__ == "__main__":
    unittest.main()
