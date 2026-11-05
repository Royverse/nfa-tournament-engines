# Forecaster — Metaculus FutureEval Bot

Autonomous forecasting engine competing in the **Metaculus FutureEval** and **MiniBench** tournaments. Generates calibrated probabilities across binary, multiple-choice, and numeric questions.

## Architecture

```
Open Questions (FutureEval + MiniBench) ──► Skip already forecast
        │
        ▼
Research (Parallel)
  ├── Google-grounded search (Gemini)
  ├── Prediction market consensus (Polymarket / Manifold)
  └── Macro & asset data desk (FRED / Market stats)
        │
        ▼
5-Slot Multi-Model Ensemble
  ├── slot-1: Base-rate anchor (Gemini 3.8 Flash)
  ├── slot-2: Causal catalyst (Gemini 3.7 Flash)
  ├── slot-3: Anti-consensus adversarial (Groq gpt-oss-120b)
  ├── slot-4: Trend inertia / mean reversion (Gemini 3.6 Flash)
  └── slot-5: Synthesis adjudicator (Nemotron / Gemini 3.5 Flash)
        │
        ▼
Calibration & Aggregation
  ├── Binary: Logit-space median, capped to [0.03, 0.97]
  ├── Multiple-Choice: Log-space probability blend, min floor 0.01
  └── Numeric: CDF percentile mapping with tail widening & de-clamping
        │
        ▼
Publish to Metaculus + Post Transparent Rationale
```

## Inference Engine & Tiers

> **Decided — no Metaculus credits.** Metaculus told the operator by email (received 2026-11-05) that it will not provide sponsored model credits at this stage. The forecaster runs on free tiers only. Do not suggest applying for credits or setting `FORECASTER_OPENROUTER_KEY`; plan improvements around the free stack. The funded path stays in the code only for the case where the operator chooses to pay for a key themselves.

| Tier | Trigger | Ensemble | Cost |
|---|---|---|---|
| **Zero-Cost Free (Active)** | `GEMINI_API_KEY`, `GROQ_API_KEY`, `OPENROUTER_FREE_KEY` | Gemini 3.8 / 3.7 / 3.6 / 3.5 Flash + Groq + OpenRouter `:free` | $0.00 |
| **User-Funded (Optional)** | `FORECASTER_OPENROUTER_KEY` | Frontier models (GPT / Claude / Gemini Pro) | User balance |

### Free-Tier Rate Pacing & Fallback
To avoid provider HTTP 429s while maintaining 100% tournament coverage:
- **Pacing**: Gemini at 4 RPM, Groq at 2 RPM, OpenRouter free at 8 RPM.
- **Dynamic Fallbacks**: If any endpoint hits a daily quota or error, the slot automatically falls back through Gemma, Groq, and OpenRouter `:free` alternatives.

## GitHub Actions Workflows

- **`Forecaster — run`**: Main production cron (every 30 minutes). Fetches open questions and submits calibrated forecasts.
- **`Forecaster — check`**: Manual diagnostics (`smoke` to verify keys; `dry-run` to test questions without publishing).
- **`Forecaster — CI`**: Runs unit test suite (`forecaster/tests/`).

## Configuration Variables

Configured via GitHub Repository Variables (`FORECASTER_*`):

| Variable | Default | Description |
|---|---|---|
| `FORECASTER_MODELS` | Default ensemble | Comma-separated LiteLLM model identifiers |
| `FORECASTER_BINARY_FLOOR` | `0.03` | Minimum probability cap for binary questions |
| `FORECASTER_BINARY_CEILING` | `0.97` | Maximum probability cap for binary questions |
| `FORECASTER_MC_FLOOR` | `0.01` | Minimum probability per multiple-choice option |
| `FORECASTER_MAX_QUESTIONS_PER_RUN` | `20` | Batch limit per scheduled execution |
| `FORECASTER_TOURNAMENTS` | `33121,minibench` | Target Metaculus tournament IDs |
| `FORECASTER_V2_ENABLED` | `false` | Enable shadow evaluation challenger bot |

## Local Development

```bash
cd forecaster
uv run pytest tests/ -q
python -m nfa_forecaster.cli smoke       # Verify API connectivity
python -m nfa_forecaster.cli test        # Dry run without submitting
```
