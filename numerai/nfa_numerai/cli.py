"""CLI entry point for the NFA Numerai Engine."""
import argparse
import sys
from .bot import run_tournament_bot
from .client import create_client, verify_credentials, get_current_round, check_submission_status
from .config import DEFAULT_MODEL_NAME, TOURNAMENT_ID

def main():
    parser = argparse.ArgumentParser(description="NFA Numerai Tournament Engine CLI")
    subparsers = parser.add_subparsers(dest="command")

    # 'run' command
    run_parser = subparsers.add_parser("run", help="Run the autonomous prediction and submission pipeline")
    run_parser.add_argument("--model", default=DEFAULT_MODEL_NAME, help="Model name slot")
    run_parser.add_argument("--force", action="store_true", help="Force upload even if round already submitted")

    # 'status' command
    status_parser = subparsers.add_parser("status", help="Check current round and model status")
    status_parser.add_argument("--model", default=DEFAULT_MODEL_NAME, help="Model name slot")

    # 'smoke' command
    subparsers.add_parser("smoke", help="Perform sanity check of credentials and API endpoints")

    args = parser.parse_args()
    cmd = args.command or "run"

    if cmd == "run":
        res = run_tournament_bot(model_name=args.model, force=args.force)
        if not res.get("success"):
            sys.exit(1)

    elif cmd == "status":
        napi = create_client()
        creds = verify_credentials(napi)
        round_num = get_current_round(napi, tournament=TOURNAMENT_ID)
        print(f"Credentials valid: {creds.get('valid')}")
        print(f"Current open round: #{round_num}")
        if round_num and creds.get("valid"):
            submitted, ref = check_submission_status(napi, round_num, model_name=args.model)
            print(f"Submitted for #{round_num}: {submitted} ({ref})")

    elif cmd == "smoke":
        napi = create_client()
        creds = verify_credentials(napi)
        round_num = get_current_round(napi, tournament=TOURNAMENT_ID)
        print(f"Smoke check:")
        print(f" - API reachable: True")
        print(f" - Credentials valid: {creds.get('valid')}")
        print(f" - Models found: {list(creds.get('models', {}).keys())}")
        print(f" - Current round: #{round_num}")
        if not creds.get("valid") or not round_num:
            sys.exit(1)

if __name__ == "__main__":
    main()
