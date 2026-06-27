"""Unit tests for layerwise formation scalars."""

from routing.layerwise import formation_scalars, slope_margin, stab_layer


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
    # transitions = [|m2-m1|, |m3-m2|, |m4-m3|] = [0.15, 0.01, 0.005]
    # first k=2 stable run starts at |m3-m2|; target layer = 3
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
    # L=1: no adjacent step — stabilization undefined; sentinel = L
    assert stab_layer([0.42], eps=0.02, k=2) == 1


def test_depth_fraction_list():
    from routing.layerwise import depth_fraction_list, trace_record, LayerTrace

    assert depth_fraction_list(16) == [i / 16 for i in range(1, 17)]
    assert depth_fraction_list(28)[0] == 1 / 28
    assert depth_fraction_list(28)[-1] == 1.0

    trace = LayerTrace(
        num_layers=4,
        margins=[0.1, 0.2, 0.21, 0.22],
        entropies=[1.0, 0.9, 0.8, 0.7],
        slope_margin=0.04,
        stabilization_layer=3,
    )
    rec = trace_record(query_id="q1", trace=trace, stab_eps=0.02, stab_k=2)
    assert "layers" not in rec
    assert rec["depth_fraction"] == [0.25, 0.5, 0.75, 1.0]
    assert rec["stabilization_frac"] == 0.75


def test_trace_depth_fraction_legacy():
    from routing.formation_analysis import trace_depth_fraction

    legacy = {"num_layers": 4, "margin": [0.1, 0.2, 0.3, 0.4]}
    assert trace_depth_fraction(legacy) == [0.25, 0.5, 0.75, 1.0]


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
