# nfa-tournament-engines

The competition bots behind [notfinancialadvice.site](https://notfinancialadvice.site).
Three Python engines, each run on GitHub Actions:

| Engine | Competition | How it runs |
|---|---|---|
| [`forecaster/`](forecaster/) | [Metaculus FutureEval](https://www.metaculus.com/) bot tournament + MiniBench | The web app's watcher dispatches `Forecaster — run` when a question opens; an hourly safety-net run backs it up |
| [`numerai/`](numerai/) | [Numerai](https://numer.ai/) Classic | Daily round submissions |
| [`crunchdao/`](crunchdao/) | [CrunchDAO](https://hub.crunchdao.com/) Synth and DataCrunch-2 | Manual `crunch push` from `CrunchDAO — run`; CrunchDAO runs the code each round |

The web app (dashboards, the 10-minute watcher, alerts) lives in a separate
private repository. It reads this repo's workflow runs and dispatches
`forecaster-run.yml` through a fine-grained token.

The forecaster runs on free model tiers only: Gemini free tier, Groq, and
OpenRouter `:free` models, with Exa, Tavily and Gemini search grounding. See
[`forecaster/README.md`](forecaster/README.md).

## Public logs

This repo is public, so its Actions logs are public. A live forecaster run sets
`FORECASTER_REDACT_LIVE=1`, which keeps forecast values and reasoning out of
the log and the run summary until each question closes. The bot's comment on
Metaculus carries the reasoning, and it becomes public when the question
closes. Secrets are passed only as Actions secrets and never printed.

## Secrets

Settings → Secrets and variables → Actions:

| Secret | Used by |
|---|---|
| `METACULUS_TOKEN` | forecaster (required) |
| `GEMINI_API_KEY`, `GROQ_API_KEY` | forecaster models and search |
| `OPENROUTER_API_KEY` | forecaster: passed to `:free` models only, so it never bills |
| `EXA_API_KEY`, `TAVILY_API_KEY` | forecaster web search |
| `ALPHA_VANTAGE_API_KEY` | forecaster stock-data fallback |
| `NUMERAI_PUBLIC_ID`, `NUMERAI_SECRET_KEY`, `NUMERAI_MCP_AUTH` | numerai |
| `CRUNCH_API_KEY` | crunchdao |

Optional: `METACULUS_TOKEN_V2` (challenger bot), `ASKNEWS_API_KEY` /
`ASKNEWS_CLIENT_ID` / `ASKNEWS_SECRET`.

## Checks

- `Forecaster — check` (Actions tab): `smoke` tests every key and service and
  forecasts nothing; `dry-run` forecasts 3 bot-testing-area questions and
  publishes nothing.
- `Forecaster — CI` runs the unit tests on every push and pull request.

## Running locally (Windows)

Put the keys in `.env.local` at the repo root (git-ignored), then:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run-forecaster.ps1 smoke
powershell -ExecutionPolicy Bypass -File scripts/run-forecaster.ps1 test 3
```

`test` forecasts bot-testing-area questions and publishes nothing unless you
add `-Publish`. The script uses [uv](https://docs.astral.sh/uv/).
