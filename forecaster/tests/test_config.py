import pytest

from nfa_forecaster.config import (
    CURRENT_TOURNAMENTS,
    FUNDED_MODELS,
    FUNDED_PARSER,
    FREE_ENSEMBLE_PREFERENCE,
    FREE_PARSER,
    GEMINI_FREE_MODELS,
    GEMINI_LITE_RESERVE,
    GEMINI_RESERVE,
    GROQ_RESERVE,
    OPENROUTER_FREE_RESERVE,
    ConfigError,
    apply_repository_variables,
    free_tier_rpm,
    is_openrouter_free,
    load_config,
    select_variant_token,
)


def test_no_keys_is_not_runnable_and_says_why():
    config = load_config({})
    assert config.funding == "none"
    assert not config.is_runnable
    assert "METACULUS_TOKEN" in config.not_runnable_reason()


def test_token_without_any_model_key_names_the_missing_model_key():
    config = load_config({"METACULUS_TOKEN": "t"})
    assert not config.is_runnable
    reason = config.not_runnable_reason()
    assert "FORECASTER_OPENROUTER_KEY" in reason and "GEMINI_API_KEY" in reason


def test_gemini_only_runs_the_free_ensemble_with_gemma_and_lite_in_reserve():
    config = load_config({"METACULUS_TOKEN": "t", "GEMINI_API_KEY": "g"})
    assert config.funding == "free"
    assert config.forecast_models == GEMINI_FREE_MODELS
    assert config.reserve_models == GEMINI_RESERVE + GEMINI_LITE_RESERVE
    assert config.parser_model == FREE_PARSER
    assert config.research_model is None
    assert config.reasoning_effort is None
    assert config.use_web_search
    assert config.is_runnable


def test_with_every_free_key_the_ensemble_mixes_three_families():
    # 2026-09-26: with only Gemini members, a spent day collapsed four of five
    # slots onto one model.
    config = load_config(
        {"METACULUS_TOKEN": "t", "GEMINI_API_KEY": "g", "GROQ_API_KEY": "q", "OPENROUTER_FREE_KEY": "f"}
    )
    assert config.funding == "free"
    assert config.forecast_models == FREE_ENSEMBLE_PREFERENCE[:5]
    assert {m.split("/")[0] for m in config.forecast_models} == {"gemini", "groq", "openrouter"}
    # Everything else is reserve, strongest first, Flash-Lite last.
    assert config.reserve_models[:2] == ("gemini/gemini-3.5-flash", "gemini/gemini-3-flash-preview")
    assert config.reserve_models[-2:] == GEMINI_LITE_RESERVE
    everything = set(GEMINI_FREE_MODELS + GEMINI_RESERVE + GROQ_RESERVE + OPENROUTER_FREE_RESERVE + GEMINI_LITE_RESERVE)
    assert set(config.forecast_models) | set(config.reserve_models) == everything
    assert not set(config.forecast_models) & set(config.reserve_models)


def test_groq_alone_is_enough_to_forecast():
    # A Google outage or a missing Gemini key must not stop the bot.
    config = load_config({"METACULUS_TOKEN": "t", "GROQ_API_KEY": "q"})
    assert config.funding == "free"
    assert config.forecast_models == GROQ_RESERVE
    assert config.parser_model == GROQ_RESERVE[0]
    assert not config.use_web_search  # grounded search needs the Gemini key
    assert config.is_runnable


def test_the_trading_openrouter_key_alone_never_makes_the_bot_funded():
    config = load_config({"METACULUS_TOKEN": "t", "OPENROUTER_FREE_KEY": "f"})
    assert config.funding == "free"
    assert all(is_openrouter_free(m) for m in config.forecast_models)


def test_free_mode_respects_the_per_model_budget():
    # The 2026-09-26 dry run lost every question to 429s: one model repeated in
    # the ensemble AND used as the parser blew its 5-requests-a-minute budget.
    config = load_config({"METACULUS_TOKEN": "t", "GEMINI_API_KEY": "g"})
    assert len(set(config.forecast_models)) == len(config.forecast_models)
    assert config.parser_model not in config.forecast_models
    assert config.parser_samples == 1
    assert config.gemini_rpm < 5


def test_caps_are_wider_for_the_weaker_free_models():
    free = load_config({"METACULUS_TOKEN": "t", "GEMINI_API_KEY": "g"})
    funded = load_config({"METACULUS_TOKEN": "t", "OPENROUTER_API_KEY": "o"})
    assert (free.binary_floor, free.binary_ceiling) == pytest.approx((0.03, 0.97))
    assert (funded.binary_floor, funded.binary_ceiling) == pytest.approx((0.02, 0.98))
    custom = load_config({"OPENROUTER_API_KEY": "o", "FORECASTER_BINARY_FLOOR": "0.01", "FORECASTER_BINARY_CEILING": "0.99"})
    assert (custom.binary_floor, custom.binary_ceiling) == (0.01, 0.99)


def test_two_questions_at_a_time_in_both_modes_for_minibench_bursts():
    assert load_config({"GEMINI_API_KEY": "g"}).max_concurrent_questions == 2
    assert load_config({"OPENROUTER_API_KEY": "o"}).max_concurrent_questions == 2


def test_funded_mode_keeps_parallelism_and_double_parsing():
    config = load_config({"METACULUS_TOKEN": "t", "OPENROUTER_API_KEY": "o"})
    assert config.max_concurrent_questions == 2
    assert config.parser_samples == 2


def test_funded_ensemble_uses_only_families_the_metaculus_credits_pay_for():
    # The sponsored key covers OpenAI, Anthropic and Google models only.
    config = load_config({"METACULUS_TOKEN": "t", "OPENROUTER_API_KEY": "o", "GEMINI_API_KEY": "g"})
    assert config.funding == "funded"
    assert config.forecast_models == FUNDED_MODELS
    for model in (*FUNDED_MODELS, FUNDED_PARSER):
        assert model.split("/")[1] in {"openai", "anthropic", "google"}, model
    assert config.parser_model == FUNDED_PARSER
    assert config.reasoning_effort == "high"
    # Perplexity is not covered by the credits, so no research model by default.
    assert config.research_model is None


def test_funded_falls_back_to_the_free_models_when_credits_fail():
    config = load_config({"METACULUS_TOKEN": "t", "OPENROUTER_API_KEY": "o", "GEMINI_API_KEY": "g"})
    assert config.reserve_models[: len(GEMINI_FREE_MODELS)] == GEMINI_FREE_MODELS
    assert not set(config.reserve_models) & set(config.forecast_models)


def test_reserve_can_be_switched_off_or_replaced():
    base = {"METACULUS_TOKEN": "t", "GEMINI_API_KEY": "g"}
    assert load_config({**base, "FORECASTER_RESERVE_MODELS": "off"}).reserve_models == ()
    assert load_config({**base, "FORECASTER_RESERVE_MODELS": "groq/a, groq/b"}).reserve_models == ("groq/a", "groq/b")
    assert load_config({**base, "FORECASTER_RESERVE_MODELS": ""}).reserve_models  # blank = automatic


def test_free_tiers_are_paced_and_paid_models_are_not():
    assert free_tier_rpm("gemini/gemini-3.6-flash", 4.0) == 4.0
    assert free_tier_rpm("groq/openai/gpt-oss-120b", 4.0) == 2.0
    assert free_tier_rpm("openrouter/qwen/qwen3.8-27b:free", 4.0) == 8.0
    assert free_tier_rpm("openrouter/openai/gpt-6-sol", 4.0) is None


def test_reasoning_effort_is_validated_and_can_be_switched_off():
    funded = {"METACULUS_TOKEN": "t", "OPENROUTER_API_KEY": "o"}
    assert load_config({**funded, "FORECASTER_REASONING_EFFORT": "off"}).reasoning_effort is None
    assert load_config({**funded, "FORECASTER_REASONING_EFFORT": "medium"}).reasoning_effort == "medium"
    with pytest.raises(ConfigError):
        load_config({**funded, "FORECASTER_REASONING_EFFORT": "extreme"})


def test_blank_keys_count_as_missing():
    # GitHub passes an unset secret as an empty string.
    config = load_config({"METACULUS_TOKEN": "  ", "OPENROUTER_API_KEY": "", "GEMINI_API_KEY": ""})
    assert config.funding == "none"
    assert not config.has_metaculus_token


def test_model_list_override_and_research_off():
    config = load_config(
        {
            "METACULUS_TOKEN": "t",
            "OPENROUTER_API_KEY": "o",
            "FORECASTER_MODELS": "openrouter/a/one, openrouter/b/two ,",
            "FORECASTER_RESEARCH_MODEL": "off",
        }
    )
    assert config.forecast_models == ("openrouter/a/one", "openrouter/b/two")
    assert config.research_model is None


def test_asknews_needs_both_oauth_halves_or_an_api_key():
    assert not load_config({"ASKNEWS_CLIENT_ID": "id"}).use_asknews
    assert load_config({"ASKNEWS_CLIENT_ID": "id", "ASKNEWS_SECRET": "s"}).use_asknews
    assert load_config({"ASKNEWS_API_KEY": "k"}).use_asknews


def test_feature_flags_default_on_and_can_be_switched_off():
    config = load_config({"GEMINI_API_KEY": "g"})
    assert config.use_markets and config.use_data_desk and config.use_web_search
    config = load_config(
        {"GEMINI_API_KEY": "g", "FORECASTER_MARKETS": "false", "FORECASTER_DATA_DESK": "0", "FORECASTER_WEB_SEARCH": "no"}
    )
    assert not (config.use_markets or config.use_data_desk or config.use_web_search)


def test_precedents_are_off_until_metaculus_shows_bots_the_outcomes():
    assert not load_config({}).use_precedents
    assert load_config({"FORECASTER_PRECEDENTS": "on"}).use_precedents


def test_unset_github_variables_arrive_blank_and_mean_default():
    # ${{ vars.X }} for an unset variable is "", which must not switch features off.
    config = load_config({"FORECASTER_MARKETS": "", "FORECASTER_DATA_DESK": "  ", "FORECASTER_MODELS": ""})
    assert config.use_markets and config.use_data_desk


def test_tournaments_default_to_the_current_season_and_can_be_overridden():
    assert load_config({}).tournaments == CURRENT_TOURNAMENTS == ("33121", "minibench")
    assert load_config({"FORECASTER_TOURNAMENTS": "spring-futureeval-2027,minibench"}).tournaments == (
        "spring-futureeval-2027",
        "minibench",
    )


@pytest.mark.parametrize(
    "env",
    [
        {"FORECASTER_BINARY_FLOOR": "0"},
        {"FORECASTER_BINARY_CEILING": "1"},
        {"FORECASTER_BINARY_FLOOR": "0.6", "FORECASTER_BINARY_CEILING": "0.4"},
        {"FORECASTER_MC_FLOOR": "0.5"},
        {"FORECASTER_MAX_CONCURRENT_QUESTIONS": "0"},
        {"FORECASTER_BINARY_FLOOR": "abc"},
        {"FORECASTER_PARSER_SAMPLES": "0"},
        {"FORECASTER_GEMINI_RPM": "0"},
    ],
)
def test_out_of_range_settings_are_rejected(env):
    with pytest.raises(ConfigError):
        load_config(env)


def test_repository_variables_are_applied_and_v2_overrides_only_for_v2():
    vars_json = '{"FORECASTER_BINARY_FLOOR": "0.05", "FORECASTER_V2_BINARY_FLOOR": "0.02", "UNRELATED": "x", "FORECASTER_VARIANT": "v2"}'
    v1_env = {"FORECASTER_VARIANT": "v1"}
    applied = apply_repository_variables(v1_env, vars_json)
    assert v1_env["FORECASTER_BINARY_FLOOR"] == "0.05"
    assert "UNRELATED" not in v1_env and v1_env["FORECASTER_VARIANT"] == "v1"
    assert applied == ["FORECASTER_BINARY_FLOOR"]

    v2_env = {"FORECASTER_VARIANT": "v2"}
    apply_repository_variables(v2_env, vars_json)
    assert v2_env["FORECASTER_BINARY_FLOOR"] == "0.02"


def test_repository_variables_reject_garbage_and_ignore_empty():
    assert apply_repository_variables({}, "") == []
    assert apply_repository_variables({}, "{}") == []
    with pytest.raises(ConfigError):
        apply_repository_variables({}, "not json")
    with pytest.raises(ConfigError):
        apply_repository_variables({}, "[1, 2]")


def test_v2_never_borrows_the_prize_bot_token():
    env = {"FORECASTER_VARIANT": "v2", "METACULUS_TOKEN": "v1-token"}
    select_variant_token(env)
    assert env["METACULUS_TOKEN"] == ""
    env = {"FORECASTER_VARIANT": "v2", "METACULUS_TOKEN": "v1-token", "METACULUS_TOKEN_V2": "v2-token"}
    select_variant_token(env)
    assert env["METACULUS_TOKEN"] == "v2-token"
    env = {"METACULUS_TOKEN": "v1-token", "METACULUS_TOKEN_V2": "v2-token"}
    select_variant_token(env)
    assert env["METACULUS_TOKEN"] == "v1-token"


def test_describe_never_contains_key_values():
    secret = "sk-or-v1-supersecretvalue"
    config = load_config(
        {
            "METACULUS_TOKEN": "metaculus-secret-token",
            "OPENROUTER_API_KEY": secret,
            "GEMINI_API_KEY": "AIzaSecret",
            "GROQ_API_KEY": "gsk_secret",
            "OPENROUTER_FREE_KEY": "sk-or-free-secret",
        }
    )
    text = "\n".join(config.describe())
    for value in (secret, "metaculus-secret-token", "AIzaSecret", "gsk_secret", "sk-or-free-secret"):
        assert value not in text
