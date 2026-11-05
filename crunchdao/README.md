# NFA CrunchDAO Tournament Engine

Autonomous quantitative machine learning engine competing in **CrunchDAO Data Science Tournaments (DataCrunch & Synth)**.

Model slot: **`royverse`**  
Target competition: **`datacrunch`** (Global Equity Forecasting)  
Payout currency: **Direct USDC** (Zero personal token staking required)

---

## 🎯 Strategic Value & Why CrunchDAO

1. **Zero-Capital Direct Cash Payouts**:
   - In Numerai Classic, you must stake NMR to earn rewards, exposing capital to drawdown burns.
   - **CrunchDAO distributes direct USDC prizes without requiring staking.**
   - Annual prize pools: **60,000 USDC/year on Target B**, **20,000 USDC/year on Target G**, plus Grand Challenges ($100k+).
   - Scored on Spearman rank correlation against market realizations.

2. **Self-Healing Autonomous Lifecycle**:
   - Implements CrunchDAO's standard `train()` and `infer()` interface in `main.py`.
   - Uses LightGBM rank-normalized regressions preserving relative stock percentile ordering within `[0.0, 1.0]`.
   - Caches trained model weights in `resources/` to conserve GitHub Actions runner compute.

3. **Automation & Schedule**:
   - GitHub Actions workflow [`.github/workflows/crunchdao-run.yml`](../.github/workflows/crunchdao-run.yml) executes:
     - **Friday 16:30 UTC**: Weekly round opening and data release.
     - **Saturday 10:00 UTC**: Weekly verification and submission safety-net.
     - **On-demand**: Manual triggers via GitHub Actions `workflow_dispatch`.

4. **Secrets & Environment Variables**:
   - `CRUNCHDAO_TOKEN`: Your API token generated from [CrunchDAO Hub](https://hub.crunchdao.com/) on the competition Submit tab.
   - `CRUNCHDAO_COMPETITION_NAME`: Defaults to `datacrunch`.
   - `CRUNCHDAO_MODEL_NAME`: Defaults to `royverse`.

---

## 💻 CLI Usage

From the project root or `crunchdao/` directory:

```bash
# Verify credentials and CLI installation
python -m nfa_crunchdao.cli smoke

# Run local benchmark validation
python -m nfa_crunchdao.cli test

# Run full autonomous setup, test, and push pipeline
python -m nfa_crunchdao.cli run --competition datacrunch --model royverse

# Inspect engine status
python -m nfa_crunchdao.cli status
```

---

## 🏗️ Architecture

```
crunchdao/
├── main.py                     # Standard CrunchDAO entry point: train() & infer()
├── requirements.txt            # Python dependencies (crunch-cli, lightgbm, etc.)
├── resources/                  # Serialized model weights (lgb_crunchdao_model.joblib)
├── data/                       # Local cached datasets
├── nfa_crunchdao/
│   ├── __init__.py
│   ├── config.py               # Hyperparameters, paths, and defaults
│   ├── client.py               # Token discovery, crunch-cli wrapper, retries
│   ├── model.py                # LightGBM rank model, Spearman metric, feature filters
│   ├── bot.py                  # End-to-end autonomous runner & push orchestrator
│   └── cli.py                  # Multi-command CLI interface
└── tests/
    └── test_crunchdao_engine.py # Unit tests
```
