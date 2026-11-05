from nfa_forecaster import cli


def test_unconfigured_run_is_a_clean_skip_not_a_failure(monkeypatch, tmp_path, capsys):
    for name in ("METACULUS_TOKEN", "OPENROUTER_API_KEY", "GEMINI_API_KEY", "GROQ_API_KEY", "OPENROUTER_FREE_KEY"):
        monkeypatch.delenv(name, raising=False)
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    assert cli.main(["run"]) == 0
    assert "Not configured yet" in summary.read_text(encoding="utf-8")


def test_bad_setting_fails_loudly(monkeypatch, tmp_path):
    monkeypatch.setenv("FORECASTER_BINARY_FLOOR", "2")
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(tmp_path / "s.md"))
    assert cli.main(["run"]) == 1
    assert "Configuration error" in (tmp_path / "s.md").read_text(encoding="utf-8")


def test_error_outcome_recovers_the_question_url_and_title():
    error = RuntimeError("Error while processing question url: 'https://www.metaculus.com/questions/4242': boom")
    outcome = cli._outcome_from_error(error, {"https://www.metaculus.com/questions/4242": "Will it?"})
    assert outcome.url == "https://www.metaculus.com/questions/4242"
    assert outcome.title == "Will it?"
    assert outcome.error.startswith("RuntimeError: ")


def test_a_redacted_live_run_keeps_forecasts_out_of_the_public_log(monkeypatch, tmp_path):
    import logging

    from nfa_forecaster.config import load_config
    from nfa_forecaster.summary import QuestionOutcome

    monkeypatch.setenv(cli.REDACT_ENV, "1")
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    question = object()
    monkeypatch.setattr(cli, "_fetch_open_questions", lambda tournaments: ([question], []))
    monkeypatch.setattr(cli, "select_questions", lambda qs, cap, now: (qs, 0))
    monkeypatch.setattr(cli, "_forecast", lambda *a, **k: (["report"], []))
    monkeypatch.setattr(
        cli,
        "_outcomes",
        lambda reports, qs: [QuestionOutcome("Will X?", "https://m/q/1", "binary", "41.0%", 0.0, 3.0, None, "why")],
    )
    forecasts = logging.getLogger("nfa_forecaster.forecasts")
    monkeypatch.setattr(forecasts, "level", logging.NOTSET)

    assert cli.command_run(load_config({"METACULUS_TOKEN": "t", "GEMINI_API_KEY": "g"}), 5) == 0
    text = summary.read_text(encoding="utf-8")
    assert "41.0%" not in text and "posted (withheld)" in text
    assert not forecasts.isEnabledFor(logging.INFO)
