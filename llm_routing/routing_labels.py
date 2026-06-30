"""Oracle routing labels — shared by scorecard and signal analysis."""

from __future__ import annotations


def routing_bucket_name(low_model_correct: int, high_model_correct: int) -> str:
    if low_model_correct and high_model_correct:
        return "easy"
    if not low_model_correct and high_model_correct:
        return "opportunity"
    if low_model_correct and not high_model_correct:
        return "lo_only"
    return "too_hard"


def routing_oracle_r(low_model_correct: int, high_model_correct: int) -> int:
    """r(q)=1 iff M_lo failed and M_hi succeeded (opportunity bucket)."""
    return int(low_model_correct == 0 and high_model_correct == 1)
