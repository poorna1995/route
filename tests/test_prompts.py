from llm_routing.corpus import Query
from llm_routing.oracle import build_messages, grade_query, parse_answer
from llm_routing.setting import get_protocol, load_setting


def test_build_messages():
    setting = load_setting("experiments/candidates/arc.yaml")
    query = Query("id", "ARC-Challenge", "Q?", ("a", "b"), 0)
    messages = build_messages(query, get_protocol(setting))
    assert messages[0]["role"] == "system"
    assert parse_answer("B", 2) == 1
    assert grade_query(Query("x", "ARC", "q", ("a", "b", "c", "d"), 1), "B") == 1
