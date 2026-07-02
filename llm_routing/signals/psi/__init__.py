from llm_routing.signals.psi.protocol import (
    ARTIFACT_VERSION,
    METRICS_VERSION,
    MODEL_RESPONSE_METRIC_KEYS,
    McqLetterProtocolExtractor,
    aggregate_choice_logprobs,
    aggregate_choice_scores,
    build_protocol_artifact,
    capture_candidate_scores,
    capture_protocol_trace,
    choice_probabilities,
    get_protocol_extractor,
    has_protocol_trace,
    pack_generation_trace,
    prompt_sha256,
    trace_generated_text,
)

__all__ = [
    "ARTIFACT_VERSION",
    "METRICS_VERSION",
    "MODEL_RESPONSE_METRIC_KEYS",
    "McqLetterProtocolExtractor",
    "aggregate_choice_logprobs",
    "aggregate_choice_scores",
    "build_protocol_artifact",
    "capture_candidate_scores",
    "capture_protocol_trace",
    "choice_probabilities",
    "extract_model_response_signals",
    "get_protocol_extractor",
    "has_protocol_trace",
    "pack_generation_trace",
    "prompt_sha256",
    "trace_generated_text",
]


def extract_model_response_signals(*args, **kwargs):
    from llm_routing.signals.psi.stage import extract_model_response_signals as _extract

    return _extract(*args, **kwargs)
