from nfa_forecaster.precedents import precedent_research, summarise_precedents


def _post(post_id, title, resolution, kind="binary", unit=None, when="2026-08-01T00:00:00Z"):
    question = {"title": title, "type": kind, "resolution": resolution, "actual_resolve_time": when}
    if unit:
        question["unit"] = unit
    return {"id": post_id, "title": title, "question": question}


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def test_summary_lists_outcomes_and_a_yes_no_base_rate():
    posts = [
        _post(1, "Will AAPL close above $250 on Aug 1?", "yes"),
        _post(2, "Will AAPL close above $260 on Aug 8?", "no"),
        _post(3, "Will AAPL close above $255 on Aug 15?", "annulled"),
        _post(4, "How many new forecasters per day?", "4.2", kind="numeric", unit="forecasters"),
        {"id": 5, "title": "A group post", "group_of_questions": {}},
        _post(99, "The question itself", "yes"),
    ]
    text = summarise_precedents(posts, exclude_post_id=99)
    assert '- "Will AAPL close above $250 on Aug 1?" → resolved YES (2026-08-01)' in text
    assert '- "How many new forecasters per day?" → resolved 4.2 forecasters (2026-08-01)' in text
    assert "Yes/no base rate among these: 1 of 2 resolved Yes." in text
    assert "$255" not in text  # annulled questions are left out
    assert "The question itself" not in text and "A group post" not in text


def test_nothing_usable_adds_nothing_to_the_prompt():
    assert summarise_precedents([_post(1, "x", "ambiguous")], exclude_post_id=2) == ""
    assert precedent_research(2, "t", "TOKEN", get=lambda url, **kw: FakeResponse({"results": []})) == ""


def test_asks_for_similar_resolved_posts_then_falls_back_to_a_title_search():
    calls = []

    def get(url, params=None, headers=None, timeout=None):
        calls.append(params)
        if "similar_to_post_id" in params:
            return FakeResponse({"results": []})  # a new post has no embedding yet
        return FakeResponse({"results": [_post(7, "Similar", "no")]})

    text = precedent_research(42, "Will the thing happen?", "TOKEN", get=get)
    assert calls[0] == {"statuses": "resolved", "limit": 12, "similar_to_post_id": 42}
    assert calls[1]["search"] == "Will the thing happen?"
    assert '"Similar" → resolved NO' in text


def test_the_token_goes_in_the_authorization_header():
    seen = {}

    def get(url, params=None, headers=None, timeout=None):
        seen.update(headers=headers, params=params)
        return FakeResponse({"results": [_post(7, "Similar", "yes")]})

    precedent_research(42, "t", "TOKEN", get=get)
    assert seen["headers"] == {"Authorization": "Token TOKEN"}
    assert "TOKEN" not in str(seen["params"])
