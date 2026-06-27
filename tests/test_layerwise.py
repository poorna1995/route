"""Unit tests for layerwise formation scalars and Route B repr drift."""

import torch

from routing.layerwise import (
    LayerTrace,
    compute_representation_drift,
    depth_fraction_list,
    drift_depth_fraction_list,
    formation_scalars,
    slope_margin,
    stab_layer,
    trace_record,
)


def _make_trace(**kwargs) -> LayerTrace:
    defaults = dict(
        num_layers=4,
        margins=[0.1, 0.2, 0.21, 0.22],
        entropies=[1.0, 0.9, 0.8, 0.7],
        slope_margin=0.04,
        stabilization_layer=3,
        adjacent_cos=[0.99, 0.995, 0.998],
        drift=[0.01, 0.005, 0.002],
        total_representation_drift=0.017,
        mean_adjacent_cos=0.994333,
        repr_adjacent_std=0.004,
    )
    defaults.update(kwargs)
    return LayerTrace(**defaults)


def test_slope_increasing():
    assert abs(slope_margin([0.1, 0.2, 0.3, 0.4]) - 0.1) < 1e-6


def test_stab_layer_stable_from_layer_three():
    m = [0.05, 0.20, 0.21, 0.215, 0.214]
    assert stab_layer(m, eps=0.02, k=2) == 3


def test_stab_layer_never_stabilizes_returns_L():
    assert stab_layer([0.1, 0.3, 0.5, 0.7], eps=0.02, k=2) == 4


def test_stab_layer_k1_single_small_step():
    m = [0.10, 0.11, 0.50]
    assert stab_layer(m, eps=0.02, k=1) == 2


def test_stab_layer_transition_semantics():
    m = [0.05, 0.20, 0.21, 0.215]
    assert stab_layer(m, eps=0.02, k=2) == 3


def test_stab_layer_never_returns_one_when_L_ge_2():
    cases = [
        [0.1, 0.2],
        [0.05, 0.20, 0.21, 0.215, 0.214],
        [0.1, 0.3, 0.5, 0.7],
    ]
    for m in cases:
        assert stab_layer(m, eps=0.02, k=2) >= 2


def test_stab_layer_no_transition_returns_L():
    assert stab_layer([0.42], eps=0.02, k=2) == 1


def test_depth_fraction_list():
    assert depth_fraction_list(16) == [i / 16 for i in range(1, 17)]
    assert depth_fraction_list(28)[0] == 1 / 28
    assert depth_fraction_list(28)[-1] == 1.0


def test_drift_depth_fraction_list():
    assert drift_depth_fraction_list(16) == [i / 16 for i in range(1, 16)]
    assert len(drift_depth_fraction_list(16)) == 15
    assert drift_depth_fraction_list(1) == []


def test_compute_representation_drift_identical_vectors():
    # hs[0]=embed; hs[1..3]=layer outputs (3 layers → 2 transitions)
    v = torch.ones(1, 4, 8)
    hs = (v.clone(), v.clone(), v.clone(), v.clone())
    adj, drift, total, mean_cos, std_d = compute_representation_drift(hs, 0, 2, 3)
    assert len(adj) == 2
    assert all(abs(c - 1.0) < 1e-6 for c in adj)
    assert all(abs(d) < 1e-6 for d in drift)
    assert abs(total) < 1e-6
    assert abs(mean_cos - 1.0) < 1e-6
    assert abs(std_d) < 1e-6


def test_compute_representation_drift_orthogonal_step():
    a = torch.tensor([1.0, 0.0])
    b = torch.tensor([0.0, 1.0])
    hs = (
        torch.zeros(1, 1, 2),
        a.view(1, 1, 2),
        b.view(1, 1, 2),
        b.view(1, 1, 2),
    )
    adj, drift, total, mean_cos, std_d = compute_representation_drift(hs, 0, 0, 3)
    assert len(adj) == 2
    assert abs(adj[0] - 0.0) < 1e-6
    assert abs(adj[1] - 1.0) < 1e-6
    assert abs(drift[0] - 1.0) < 1e-6
    assert abs(drift[1]) < 1e-6
    assert abs(total - 1.0) < 1e-6


def test_trace_record_includes_route_b():
    trace = _make_trace()
    rec = trace_record(query_id="q1", trace=trace, stab_eps=0.02, stab_k=2)
    assert "layers" not in rec
    assert rec["depth_fraction"] == [0.25, 0.5, 0.75, 1.0]
    assert rec["drift_depth_fraction"] == [0.25, 0.5, 0.75]
    assert rec["stabilization_frac"] == 0.75
    assert rec["drift"] == [0.01, 0.005, 0.002]
    assert rec["total_representation_drift"] == 0.017


def test_trace_record_nan_margin_serializes_null():
    trace = _make_trace(margins=[float("nan"), float("nan"), 0.21, 0.22])
    rec = trace_record(query_id="q1", trace=trace, stab_eps=0.02, stab_k=2)
    assert rec["margin"][0] is None
    assert rec["margin"][-1] == 0.22


def test_trace_depth_fraction_legacy():
    from routing.formation_analysis import trace_depth_fraction

    legacy = {"num_layers": 4, "margin": [0.1, 0.2, 0.3, 0.4]}
    assert trace_depth_fraction(legacy) == [0.25, 0.5, 0.75, 1.0]

    with_drift = {"num_layers": 4, "drift_depth_fraction": [0.1, 0.2, 0.3]}
    assert trace_depth_fraction(with_drift, metric="drift") == [0.1, 0.2, 0.3]


def test_prepare_output_path_requires_overwrite(tmp_path):
    from routing.layerwise import _prepare_output_path

    existing = tmp_path / "trace.jsonl"
    existing.write_text("{}\n")
    try:
        _prepare_output_path(existing, overwrite=False)
        assert False, "expected FileExistsError"
    except FileExistsError:
        pass
    _prepare_output_path(existing, overwrite=True)
    assert not existing.exists()


def test_formation_scalars_tuple():
    m = [0.10, 0.11, 0.50]
    slope, stab = formation_scalars(m, eps=0.02, k=1)
    assert stab == 2
    assert abs(slope - slope_margin(m)) < 1e-9


def test_bucket_medians_handles_null_margins(tmp_path):
    from routing.formation_analysis import bucket_medians

    traces = {
        "q1": {"margin": [None, None, 0.3, 0.4]},
        "q2": {"margin": [None, None, 0.5, 0.6]},
    }
    merged = __import__("pandas").DataFrame(
        {"query_id": ["q1", "q2"], "bucket": ["easy", "easy"]}
    )
    medians = bucket_medians(traces, merged, metric="margin")
    assert len(medians["easy"]) == 4
    assert medians["easy"][-1] == 0.5
