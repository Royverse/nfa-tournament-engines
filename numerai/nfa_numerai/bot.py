"""High-level autonomous tournament bot with self-healing and idempotency."""
import os
import sys
from typing import Dict, Any, Optional

from .config import (
    DATA_DIR,
    DATA_VERSION,
    DEFAULT_MODEL_NAME,
    DEFAULT_MODEL_ID,
    TOURNAMENT_ID,
)
from .client import (
    create_client,
    verify_credentials,
    get_current_round,
    check_submission_status,
    download_dataset_safe,
)
from .model import (
    train_or_load_model,
    train_live_distilled_ensemble,
    generate_predictions,
    MODEL_FILE,
)

def run_tournament_bot(
    model_name: str = DEFAULT_MODEL_NAME,
    force: bool = False,
    limit_features: bool = False,
) -> Dict[str, Any]:
    """Execute the full autonomous tournament workflow."""
    print("=" * 60)
    print(f"NFA Numerai Tournament Engine — Model: {model_name} (Tournament {TOURNAMENT_ID})")
    print("=" * 60)

    # 1. Check API & Credentials
    napi = create_client()
    cred_check = verify_credentials(napi)
    if not cred_check.get("valid"):
        err = cred_check.get("error", "Unknown credential failure")
        print(f"[Bot] ERROR: Credentials invalid or missing: {err}", file=sys.stderr)
        return {"success": False, "error": f"Credentials error: {err}"}

    models = cred_check.get("models", {})
    target_model_id = models.get(model_name, DEFAULT_MODEL_ID)
    print(f"[Bot] Authenticated. Model slot: {model_name} -> {target_model_id}")

    # 2. Check Current Round
    current_round = get_current_round(napi, tournament=TOURNAMENT_ID)
    if not current_round:
        print("[Bot] ERROR: Could not determine current tournament round.", file=sys.stderr)
        return {"success": False, "error": "Could not determine current round"}

    print(f"[Bot] Current open round: #{current_round}")

    # 3. Idempotency Check
    already_submitted, sub_ref = check_submission_status(napi, current_round, model_name=model_name)
    if already_submitted and not force:
        print(f"[Bot] Round #{current_round} already submitted ({sub_ref}). Skipping to conserve compute.")
        return {
            "success": True,
            "round": current_round,
            "status": "already_submitted",
            "submission_ref": sub_ref,
        }

    # 4. Data Acquisition
    live_parquet = os.path.join(DATA_DIR, f"{DATA_VERSION}_live.parquet")
    features_json = os.path.join(DATA_DIR, f"{DATA_VERSION}_features.json")
    train_parquet = os.path.join(DATA_DIR, f"{DATA_VERSION}_train.parquet")
    live_benchmark_parquet = os.path.join(DATA_DIR, f"{DATA_VERSION}_live_benchmark_models.parquet")
    live_example_parquet = os.path.join(DATA_DIR, f"{DATA_VERSION}_live_example_preds.parquet")

    if not download_dataset_safe(napi, f"{DATA_VERSION}/live.parquet", live_parquet):
        return {"success": False, "error": "Failed to download live.parquet"}

    if not download_dataset_safe(napi, f"{DATA_VERSION}/features.json", features_json):
        return {"success": False, "error": "Failed to download features.json"}

    # Best-effort download of live benchmark models for BMC/MMC orthogonalization
    download_dataset_safe(napi, f"{DATA_VERSION}/live_benchmark_models.parquet", live_benchmark_parquet)

    # 5. Model Loading or Training (with live distillation fallback if train.parquet is regenerating on Saturday)
    booster = None
    if not os.path.exists(MODEL_FILE):
        print(f"[Bot] Pre-trained model not found. Fetching training data...")
        if download_dataset_safe(napi, f"{DATA_VERSION}/train.parquet", train_parquet, max_retries=1):
            booster = train_or_load_model(train_parquet, features_json)
        else:
            print(
                f"[Bot] {DATA_VERSION}/train.parquet is currently offline for weekend round regeneration; "
                f"running multi-target live distillation ensemble..."
            )
            download_dataset_safe(
                napi, f"{DATA_VERSION}/live_example_preds.parquet", live_example_parquet, max_retries=1
            )
            booster = train_live_distilled_ensemble(
                live_parquet_path=live_parquet,
                features_json_path=features_json,
                live_benchmark_parquet_path=live_benchmark_parquet if os.path.exists(live_benchmark_parquet) else None,
                live_example_parquet_path=live_example_parquet if os.path.exists(live_example_parquet) else None,
            )
    else:
        booster = train_or_load_model(train_parquet, features_json)

    # 6. Generate Live Predictions
    _, submission_path = generate_predictions(
        booster,
        live_parquet,
        features_json,
        live_benchmark_parquet_path=live_benchmark_parquet if os.path.exists(live_benchmark_parquet) else None,
    )

    # 7. Upload to Numerai
    print(f"[Bot] Submitting {submission_path} to model {model_name} ({target_model_id})...")
    try:
        sub_id = napi.upload_predictions(submission_path, model_id=target_model_id)
        print(f"[Bot] [SUCCESS] Predictions uploaded! Submission ID: {sub_id}")
    except Exception as e:
        print(f"[Bot] ERROR: Upload failed: {e}", file=sys.stderr)
        return {"success": False, "error": f"Upload failed: {e}"}

    # 8. Post-Submission Verification
    verified, ref = check_submission_status(napi, current_round, model_name=model_name)
    if verified:
        print(f"[Bot] [VERIFIED] Round #{current_round} confirmed active on Numerai!")
    else:
        print(f"[Bot] [NOTICE] Submission uploaded; waiting for Numerai backend to index.")

    return {
        "success": True,
        "round": current_round,
        "submission_id": sub_id,
        "verified": verified,
    }
