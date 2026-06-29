from llm_routing.model_response.protocol import (
    ARTIFACT_VERSION,
    METRICS_VERSION,
    MODEL_RESPONSE_METRIC_KEYS,
    McqLetterProtocolExtractor,
    aggregate_choice_logprobs,
    build_protocol_artifact,
    capture_protocol_trace,
    get_protocol_extractor,
    has_protocol_trace,
    mock_protocol_trace,
    pack_generation_trace,
    prompt_sha256,
)

__all__ = [
    "ARTIFACT_VERSION",
    "METRICS_VERSION",
    "MODEL_RESPONSE_METRIC_KEYS",
    "McqLetterProtocolExtractor",
    "aggregate_choice_logprobs",
    "build_protocol_artifact",
    "capture_protocol_trace",
    "extract_model_response_signals",
    "get_protocol_extractor",
    "has_protocol_trace",
    "mock_protocol_trace",
    "pack_generation_trace",
    "prompt_sha256",
]


def extract_model_response_signals(*args, **kwargs):
    from llm_routing.model_response.stage import extract_model_response_signals as _extract

    return _extract(*args, **kwargs)
