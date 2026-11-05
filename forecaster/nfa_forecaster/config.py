"""Runtime configuration, decided once from the environment.

Everything that differs between a funded run and a free-fallback run is
decided here, in one pure function, so it can be tested and so every run can
state in its log exactly which mode it ran in. Nothing in this module reads a
key's value beyond checking that it is present, and nothing here logs one.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Mapping, MutableMapping

# Funded ensemble — optional frontier models used when OPENROUTER_API_KEY is present.
# Metaculus will not provide sponsored credits (email to the operator, 2026-11-05),
# so this runs only if the operator pays for a key; the free ensemble below is the plan.
FUNDED_MODELS: tuple[str, ...] = (
    "openrouter/openai/gpt-6-sol",
    "openrouter/anthropic/claude-sonnet-5",
    "openrouter/google/gemini-3.8-flash",
    "openrouter/openai/gpt-6-sol",
    "openrouter/anthropic/claude-sonnet-5",
)
FUNDED_REASONING_EFFORT = "high"
FUNDED_PARSER = "openrouter/openai/gpt-6-luna"

# Free ensemble — Google AI Studio's free tier. Measured 2026-09-26: Flash
# models answer, Pro models return 429, and each model allows 5 requests a
# minute and 20 a DAY. Five different models, each with its own budget.
GEMINI_FREE_MODELS: tuple[str, ...] = (
    "gemini/gemini-3.8-flash",
    "gemini/gemini-3.7-flash",
    "gemini/gemini-3.6-flash",
    "gemini/gemini-3.5-flash",
    "gemini/gemini-3-flash-preview",
)
FREE_PARSER = "gemini/gemini-3.5-flash-lite"

# The free ensemble mixes three families by design, whenever the keys exist.
# The 2026-09-26 dry run showed why: with three Gemini models out for the
# day, four of five slots fell back to the same model, so the "ensemble" was
# one model's opinion. Mixing also spreads the load over three providers'
# daily limits.
FREE_ENSEMBLE_PREFERENCE: tuple[str, ...] = (
    "gemini/gemini-3.8-flash",
    "gemini/gemini-3.7-flash",
    "groq/openai/gpt-oss-120b",
    "gemini/gemini-3.6-flash",
    "openrouter/nvidia/nemotron-3-ultra-550b-a55b:free",
    "gemini/gemini-3.5-flash",
    "gemini/gemini-3-flash-preview",
)
FREE_ENSEMBLE_SIZE = 5

# The reserve: models a call falls back to when its own model is out (see
# models.py). Strongest first. Each group is used only when its key exists.
# Gemma runs on the Gemini key with its own quota; Groq's free tier and
# OpenRouter's ":free" models are separate providers entirely, so a Google
# outage or a spent Google day does not stop the bot.
GEMINI_RESERVE: tuple[str, ...] = (
    "gemini/gemma-4-31b-it",
    "gemini/gemini-2.5-flash",
    "gemini/gemma-4-26b-a4b-it",
)
# Groq's whole chat catalogue on 2026-09-26 (the smoke test lists it).
GROQ_RESERVE: tuple[str, ...] = (
    "groq/openai/gpt-oss-120b",
    "groq/qwen/qwen3.8-27b",
    "groq/openai/gpt-oss-20b",
)
OPENROUTER_FREE_RESERVE: tuple[str, ...] = (
    "openrouter/nvidia/nemotron-3-ultra-550b-a55b:free",
    "openrouter/thinkingmachines/inkling:free",
    "openrouter/qwen/qwen3.8-27b:free",
)
# Last resort: small models, good enough to parse an answer, weak forecasters.
GEMINI_LITE_RESERVE: tuple[str, ...] = (
    "gemini/gemini-3.5-flash-lite",
    "gemini/gemini-3.1-flash-lite",
)

# Requests per minute per model on each free tier, with margin: Gemini allows
# 5; Groq's limit that bites is tokens per minute (a research-heavy prompt is
# several thousand tokens); OpenRouter's free models allow 20.
DEFAULT_GEMINI_RPM = 4.0
GROQ_RPM = 2.0
OPENROUTER_FREE_RPM = 8.0


def is_openrouter_free(model: str) -> bool:
    return model.startswith("openrouter/") and model.endswith(":free")


def free_tier_rpm(model: str, gemini_rpm: float) -> float | None:
    """Pacing for a free-tier model, or None for a paid one (never paced)."""
    if model.startswith("gemini/"):
        return gemini_rpm
    if model.startswith("groq/"):
        return GROQ_RPM
    if is_openrouter_free(model):
        return OPENROUTER_FREE_RPM
    return None

# The season's tournaments: FutureEval Fall 2026 (project 33121) and MiniBench.
# These equal forecasting-tools 0.3.1's CURRENT_AI_COMPETITION_ID and
# CURRENT_MINIBENCH_ID (test_bot.py fails if a library bump drifts from them).
# The web app's Netlify watcher (private repo) carries the same pair in
# src/lib/forecaster/tournaments.ts. Change both when the season moves.
# FORECASTER_TOURNAMENTS overrides either side without a code change (a repo
# variable here, a Netlify env var there).
CURRENT_TOURNAMENTS: tuple[str, ...] = ("33121", "minibench")

# Binary caps by mode. A top-15 Summer 2026 bot priced its clip on its own
# record: tightening it past 1-2% "loses in every window and era"
# (nostreambot design log). Frontier models get 2%. The free models are
# weaker and more often confidently wrong, so they keep a wider 3% margin.
DEFAULT_BINARY_FLOOR = {"funded": 0.02, "free": 0.03, "none": 0.03}
DEFAULT_MC_FLOOR = 0.01
# Two questions at a time in free mode too. The pool spans three providers,
# each model is paced on its own, and MiniBench lands most of its ~60
# questions in a round's first few days: one at a time is too slow for that.
DEFAULT_MAX_CONCURRENT_QUESTIONS = {"funded": 2, "free": 2, "none": 1}
DEFAULT_PARSER_SAMPLES = {"funded": 2, "free": 1, "none": 1}

_OFF_VALUES = {"0", "false", "off", "no", "none", ""}


class ConfigError(ValueError):
    """The environment asks for something contradictory or out of range."""


@dataclass(frozen=True)
class ForecasterConfig:
    variant: str
    has_metaculus_token: bool
    funding: str  # "funded" | "free" | "none"
    forecast_models: tuple[str, ...]
    reserve_models: tuple[str, ...]
    parser_model: str | None
    research_model: str | None
    reasoning_effort: str | None
    use_asknews: bool
    use_web_search: bool
    use_exa: bool
    use_tavily: bool
    use_precedents: bool
    use_markets: bool
    use_data_desk: bool
    tournaments: tuple[str, ...]
    binary_floor: float
    binary_ceiling: float
    mc_floor: float
    max_concurrent_questions: int
    parser_samples: int
    gemini_rpm: float

    @property
    def is_runnable(self) -> bool:
        return self.has_metaculus_token and bool(self.forecast_models) and self.parser_model is not None

    def not_runnable_reason(self) -> str | None:
        if not self.has_metaculus_token:
            return "METACULUS_TOKEN is not set — create the bot account on Metaculus and add its token as a GitHub secret."
        if not self.forecast_models:
            return (
                "No AI model is available — set GEMINI_API_KEY, GROQ_API_KEY, "
                "or FORECASTER_OPENROUTER_KEY as a GitHub secret."
            )
        if self.parser_model is None:
            return "No parser model is available — set FORECASTER_PARSER_MODEL or provide an AI key."
        return None

    def describe(self) -> list[str]:
        """Human-readable lines for the run log and the Actions summary. No secrets."""
        tournaments = ", ".join(self.tournaments)
        on = {True: "on", False: "off"}
        return [
            f"variant: {self.variant}",
            f"funding: {self.funding}",
            f"ensemble ({len(self.forecast_models)}): {', '.join(self.forecast_models) or '—'}"
            + (f" · reasoning effort {self.reasoning_effort}" if self.reasoning_effort else ""),
            f"fallback reserve ({len(self.reserve_models)}): {', '.join(self.reserve_models) or 'off'}",
            f"parser: {self.parser_model or '—'}",
            f"research model: {self.research_model or 'off'}",
            f"AskNews: {on[self.use_asknews]} · Web search (Exa/Tavily/Gemini): {on[self.use_web_search]} (Exa: {on[self.use_exa]}, Tavily: {on[self.use_tavily]})"
            f" · Metaculus precedents: {on[self.use_precedents]} · markets: {on[self.use_markets]}"
            f" · data desk: {on[self.use_data_desk]}",
            f"binary caps: [{self.binary_floor:.2f}, {self.binary_ceiling:.2f}] · multiple-choice floor: {self.mc_floor:.2f}",
            f"tournaments: {tournaments}",
            f"max concurrent questions: {self.max_concurrent_questions} · parser samples: {self.parser_samples}"
            f" · Gemini pacing: {self.gemini_rpm:g} requests/min per model",
        ]


def _present(env: Mapping[str, str], name: str) -> bool:
    return bool((env.get(name) or "").strip())


def _flag(env: Mapping[str, str], name: str, default: bool) -> bool:
    # GitHub passes an unset repository variable as "", which must mean
    # "use the default", never "off".
    raw = (env.get(name) or "").strip().lower()
    if not raw:
        return default
    return raw not in _OFF_VALUES


def _float(env: Mapping[str, str], name: str, default: float) -> float:
    raw = (env.get(name) or "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError as e:
        raise ConfigError(f"{name} must be a number, got {raw!r}") from e


def _list(env: Mapping[str, str], name: str) -> tuple[str, ...] | None:
    raw = (env.get(name) or "").strip()
    if not raw:
        return None
    return tuple(item.strip() for item in raw.split(",") if item.strip())


def _optional_model(env: Mapping[str, str], name: str, default: str | None) -> str | None:
    raw = env.get(name)
    if raw is None or not raw.strip():
        return default
    return None if raw.strip().lower() in _OFF_VALUES else raw.strip()


_V2_PREFIX = "FORECASTER_V2_"


def apply_repository_variables(env: MutableMapping[str, str], vars_json: str | None) -> list[str]:
    """Copy FORECASTER_* GitHub repository variables into the environment.

    The workflow passes the whole `vars` context as JSON, so any tuning knob
    (models, caps, feature flags, tournaments) can be changed in the GitHub UI
    without a code change or a new line of YAML. For the v2 challenger bot,
    FORECASTER_V2_<NAME> overrides FORECASTER_<NAME>. Returns the names set.
    Repository variables are plain configuration; secrets never travel here.
    """
    if not vars_json or not vars_json.strip():
        return []
    try:
        data = json.loads(vars_json)
    except json.JSONDecodeError as e:
        raise ConfigError("FORECASTER_VARS_JSON is not valid JSON") from e
    if not isinstance(data, dict):
        raise ConfigError("FORECASTER_VARS_JSON must be a JSON object")

    applied: list[str] = []
    for name, value in data.items():
        if name.startswith("FORECASTER_") and not name.startswith(_V2_PREFIX) and name != "FORECASTER_VARIANT":
            env[name] = str(value)
            applied.append(name)
    if (env.get("FORECASTER_VARIANT") or "v1").strip() == "v2":
        for name, value in data.items():
            if name.startswith(_V2_PREFIX):
                target = "FORECASTER_" + name[len(_V2_PREFIX):]
                env[target] = str(value)
                applied.append(target)
    return applied


def select_variant_token(env: MutableMapping[str, str]) -> None:
    """Make METACULUS_TOKEN the token of the bot variant being run.

    The v2 challenger must forecast as its own account, never as the
    prize-eligible v1 bot: with no METACULUS_TOKEN_V2 it gets an empty token
    and skips, rather than falling back to v1's.
    """
    if (env.get("FORECASTER_VARIANT") or "v1").strip() == "v2":
        env["METACULUS_TOKEN"] = env.get("METACULUS_TOKEN_V2") or ""


def free_models(env: Mapping[str, str]) -> tuple[str, ...]:
    """Every free-tier model whose key is present, strongest first."""
    has_gemini = _present(env, "GEMINI_API_KEY")
    models: list[str] = []
    if has_gemini:
        models += [*GEMINI_FREE_MODELS, *GEMINI_RESERVE]
    if _present(env, "GROQ_API_KEY"):
        models += GROQ_RESERVE
    if _present(env, "OPENROUTER_FREE_KEY"):
        models += OPENROUTER_FREE_RESERVE
    if has_gemini:
        models += GEMINI_LITE_RESERVE
    return tuple(models)


def load_config(env: Mapping[str, str]) -> ForecasterConfig:
    has_openrouter = _present(env, "OPENROUTER_API_KEY")
    free = free_models(env)

    if has_openrouter:
        funding = "funded"
        default_models: tuple[str, ...] = FUNDED_MODELS
        default_parser: str | None = FUNDED_PARSER
        effort: str | None = FUNDED_REASONING_EFFORT
    elif free:
        funding = "free"
        preferred = [m for m in FREE_ENSEMBLE_PREFERENCE if m in free]
        default_models = tuple(dict.fromkeys([*preferred, *free]))[:FREE_ENSEMBLE_SIZE]
        default_parser = FREE_PARSER if FREE_PARSER in free else free[0]
        effort = None
    else:
        funding = "none"
        default_models = ()
        default_parser = None
        effort = None

    models = _list(env, "FORECASTER_MODELS") or default_models
    # "off" turns fallback off; a list replaces the automatic reserve.
    reserve_setting = (env.get("FORECASTER_RESERVE_MODELS") or "").strip()
    if reserve_setting.lower() in _OFF_VALUES - {""}:
        reserve: tuple[str, ...] = ()
    else:
        reserve = _list(env, "FORECASTER_RESERVE_MODELS") or tuple(m for m in free if m not in models)
    binary_floor = _float(env, "FORECASTER_BINARY_FLOOR", DEFAULT_BINARY_FLOOR[funding])
    binary_ceiling = _float(env, "FORECASTER_BINARY_CEILING", 1 - DEFAULT_BINARY_FLOOR[funding])
    mc_floor = _float(env, "FORECASTER_MC_FLOOR", DEFAULT_MC_FLOOR)
    concurrency = int(_float(env, "FORECASTER_MAX_CONCURRENT_QUESTIONS", DEFAULT_MAX_CONCURRENT_QUESTIONS[funding]))
    parser_samples = int(_float(env, "FORECASTER_PARSER_SAMPLES", DEFAULT_PARSER_SAMPLES[funding]))
    gemini_rpm = _float(env, "FORECASTER_GEMINI_RPM", DEFAULT_GEMINI_RPM)

    # Metaculus rejects binary forecasts outside [0.001, 0.999].
    if not 0.001 <= binary_floor < binary_ceiling <= 0.999:
        raise ConfigError(
            f"binary caps must satisfy 0.001 <= floor < ceiling <= 0.999, got [{binary_floor}, {binary_ceiling}]"
        )
    if not 0 <= mc_floor < 0.2:
        raise ConfigError(f"FORECASTER_MC_FLOOR must be in [0, 0.2), got {mc_floor}")
    if concurrency < 1:
        raise ConfigError(f"FORECASTER_MAX_CONCURRENT_QUESTIONS must be >= 1, got {concurrency}")
    if parser_samples < 1:
        raise ConfigError(f"FORECASTER_PARSER_SAMPLES must be >= 1, got {parser_samples}")
    if gemini_rpm <= 0:
        raise ConfigError(f"FORECASTER_GEMINI_RPM must be positive, got {gemini_rpm}")
    reasoning_effort = _optional_model(env, "FORECASTER_REASONING_EFFORT", effort)
    if reasoning_effort not in (None, "low", "medium", "high"):
        raise ConfigError(f"FORECASTER_REASONING_EFFORT must be low, medium, high or off, got {reasoning_effort!r}")

    return ForecasterConfig(
        variant=(env.get("FORECASTER_VARIANT") or "v1").strip() or "v1",
        has_metaculus_token=_present(env, "METACULUS_TOKEN"),
        funding=funding,
        forecast_models=models,
        reserve_models=reserve,
        parser_model=_optional_model(env, "FORECASTER_PARSER_MODEL", default_parser),
        # Off by default: Perplexity is not covered by Metaculus's credits.
        # Google-grounded search (free) does the web research instead.
        research_model=_optional_model(env, "FORECASTER_RESEARCH_MODEL", None),
        reasoning_effort=reasoning_effort,
        use_asknews=(
            (_present(env, "ASKNEWS_CLIENT_ID") and _present(env, "ASKNEWS_SECRET"))
            or _present(env, "ASKNEWS_API_KEY")
        ),
        use_web_search=(_present(env, "GEMINI_API_KEY") or _present(env, "EXA_API_KEY") or _present(env, "TAVILY_API_KEY")) and _flag(env, "FORECASTER_WEB_SEARCH", True),
        use_exa=_present(env, "EXA_API_KEY"),
        use_tavily=_present(env, "TAVILY_API_KEY"),
        # Off by default: Metaculus returns resolved questions to the bot
        # account with every outcome empty (measured 2026-09-26), so there is
        # nothing to learn from yet. The smoke test re-checks every run.
        use_precedents=_flag(env, "FORECASTER_PRECEDENTS", False),
        use_markets=_flag(env, "FORECASTER_MARKETS", True),
        use_data_desk=_flag(env, "FORECASTER_DATA_DESK", True),
        tournaments=_list(env, "FORECASTER_TOURNAMENTS") or CURRENT_TOURNAMENTS,
        binary_floor=binary_floor,
        binary_ceiling=binary_ceiling,
        mc_floor=mc_floor,
        max_concurrent_questions=concurrency,
        parser_samples=parser_samples,
        gemini_rpm=gemini_rpm,
    )
