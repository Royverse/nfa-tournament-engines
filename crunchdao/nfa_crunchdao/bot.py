"""High-level autonomous CrunchDAO tournament bot with self-healing, testing, and push automation."""
import os
import sys
from typing import Dict, Any, Optional

from .config import (
    DEFAULT_COMPETITION_NAME,
    DEFAULT_MODEL_NAME,
    RESOURCES_DIR,
)
from .client import (
    verify_credentials,
    get_auth_token,
    check_cli_available,
    setup_competition_workspace,
    test_local_submission,
    push_submission,
)

def run_tournament_bot(
    competition: str = DEFAULT_COMPETITION_NAME,
    model_name: str = DEFAULT_MODEL_NAME,
    force: bool = False,
    skip_test: bool = False,
) -> Dict[str, Any]:
    """Execute the full autonomous CrunchDAO tournament push and submission workflow."""
    print("=" * 60)
    print(f"NFA CrunchDAO Engine — Competition: {competition} | Model: {model_name}")
    print("Payout Currency: USDC (Zero-Capital Staking Free)")
    print("=" * 60)

    # 1. Check API & Credentials
    cred_check = verify_credentials()
    if not cred_check.get("valid"):
        err = cred_check.get("error", "Unknown credential failure")
        print(f"[CrunchDAO Bot] ERROR: Credentials invalid or missing: {err}", file=sys.stderr)
        return {"success": False, "error": f"Credentials error: {err}"}

    token = get_auth_token()
    print("[CrunchDAO Bot] Token found in environment.")

    # 2. Check crunch-cli
    if not check_cli_available():
        print("[CrunchDAO Bot] WARNING: crunch-cli binary not found on PATH.", file=sys.stderr)
        print("[CrunchDAO Bot] Install via: pip install crunch-cli", file=sys.stderr)
        return {
            "success": False,
            "error": "crunch-cli binary missing. Run 'pip install crunch-cli'",
        }

    # 3. Setup / Link Workspace
    print(f"[CrunchDAO Bot] Setting up workspace for competition '{competition}' with model '{model_name}'...")
    setup_res = setup_competition_workspace(competition=competition, model_name=model_name, token=token)
    if not setup_res.get("success"):
        print(f"[CrunchDAO Bot] Workspace setup warning/error: {setup_res.get('stderr')}", file=sys.stderr)
        if not force:
            return {"success": False, "error": f"Workspace setup failed: {setup_res.get('stderr')}"}

    # 4. Local Test Validation (Smoke & Mock Evaluation)
    if not skip_test:
        print("[CrunchDAO Bot] Executing local benchmark verification (`crunch test`)...")
        test_res = test_local_submission()
        if not test_res.get("success"):
            print(f"[CrunchDAO Bot] Local test failed:\n{test_res.get('stderr')}", file=sys.stderr)
            if not force:
                return {"success": False, "error": f"Local verification test failed: {test_res.get('stderr')}"}
        else:
            print("[CrunchDAO Bot] [PASSED] Local test verified successfully.")

    # 5. Push Code & Model to CrunchDAO Hub
    msg = f"Autonomous submission: {model_name} on {competition}"
    print(f"[CrunchDAO Bot] Pushing code & model to CrunchDAO Hub ('{msg}')...")
    push_res = push_submission(message=msg)
    if not push_res.get("success"):
        print(f"[CrunchDAO Bot] Push failed:\n{push_res.get('stderr')}", file=sys.stderr)
        return {"success": False, "error": f"Push failed: {push_res.get('stderr')}"}

    print(f"[CrunchDAO Bot] [SUCCESS] Model and pipeline successfully pushed to CrunchDAO Hub!")
    print(push_res.get("stdout", ""))

    return {
        "success": True,
        "competition": competition,
        "model_name": model_name,
        "payout_currency": "USDC",
        "output": push_res.get("stdout"),
    }
