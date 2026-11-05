# NFA Numerai Tournament Engine

Autonomous quantitative machine learning engine competing in **Numerai Classic (Tournament 8)**.

Model slot: **`royverse`** (ID: `43d6a18d-95e4-4c98-9e3d-713ec5f67cf1`)

---

## Architecture & Autonomous Lifecycle

1. **Zero-Capital Quantitative Hedge Fund Model**:
   - Competes in predicting 20-day stock returns across thousands of obfuscated global equities.
   - Predictions are ensembled by Numerai into the Meta-Model.
   - Scored on Correlation (`corr20`), Meta-Model Contribution (`mmc`), and Feature-Neutral Correlation (`fnc`).

2. **Self-Healing & Idempotency**:
   - The engine checks whether the current round has already received valid predictions from `royverse`.
   - If already submitted, it skips redundant execution, saving compute and quota.
   - If an upload fails, exponential backoff retries ensure delivery within the round window.
   - Strict validation verifies: non-empty rows, no NaN/Inf, strict [0, 1] percentile ranking.

3. **Automation & Schedule**:
   - GitHub Actions workflow `.github/workflows/numerai-run.yml` executes:
     - Every Saturday at 18:30 UTC (weekend round opening).
     - Every Tuesday at 14:00 UTC (mid-week round opening).
     - Daily safety-net check at 06:15 UTC.
     - Manual on-demand triggers via GitHub Actions `workflow_dispatch`.

4. **Secrets & Environment Variables**:
   - `NUMERAI_PUBLIC_ID`: API Key Public ID
   - `NUMERAI_SECRET_KEY`: API Key Secret Key
   - `NUMERAI_MCP_AUTH`: Alternatively formatted as `Token PUBLIC_ID$SECRET_KEY`
   - `NUMERAI_MODEL_NAME`: Defaults to `royverse`
