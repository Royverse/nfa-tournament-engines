"""CLI entry point for the NFA CrunchDAO Engine."""
import argparse
import sys
from .bot import run_tournament_bot
from .client import verify_credentials, check_cli_available, test_local_submission
from .config import DEFAULT_COMPETITION_NAME, DEFAULT_MODEL_NAME, RESOURCES_DIR, DATA_DIR

def main():
    parser = argparse.ArgumentParser(description="NFA CrunchDAO Tournament Engine CLI")
    subparsers = parser.add_subparsers(dest="command")

    # 'run' command
    run_parser = subparsers.add_parser("run", help="Run the autonomous submission pipeline")
    run_parser.add_argument("--competition", default=DEFAULT_COMPETITION_NAME, help="Target competition")
    run_parser.add_argument("--model", default=DEFAULT_MODEL_NAME, help="Model name slot")
    run_parser.add_argument("--force", action="store_true", help="Force proceed despite non-critical warnings")
    run_parser.add_argument("--skip-test", action="store_true", help="Skip local benchmark test prior to push")

    # 'test' command
    test_parser = subparsers.add_parser("test", help="Run local crunch-cli test validation")

    # 'smoke' command
    subparsers.add_parser("smoke", help="Check credentials and crunch-cli installation")

    # 'status' command
    subparsers.add_parser("status", help="Inspect local CrunchDAO engine status")

    args = parser.parse_args()
    cmd = args.command or "run"

    if cmd == "run":
        res = run_tournament_bot(
            competition=args.competition,
            model_name=args.model,
            force=args.force,
            skip_test=args.skip_test,
        )
        if not res.get("success"):
            sys.exit(1)

    elif cmd == "test":
        print("Running CrunchDAO local test suite...")
        res = test_local_submission()
        if not res.get("success"):
            print(f"Test failed:\n{res.get('stderr')}", file=sys.stderr)
            sys.exit(1)
        print(res.get("stdout"))

    elif cmd == "smoke":
        creds = verify_credentials()
        cli_ok = check_cli_available()
        print("CrunchDAO Smoke Check:")
        print(f" - Credentials valid: {creds.get('valid')}")
        if not creds.get("valid"):
            print(f" - Credential issue: {creds.get('error')}")
        print(f" - crunch-cli binary available: {cli_ok}")
        print(f" - Resources dir: {RESOURCES_DIR}")
        print(f" - Data dir: {DATA_DIR}")
        if not creds.get("valid") or not cli_ok:
            sys.exit(1)

    elif cmd == "status":
        creds = verify_credentials()
        cli_ok = check_cli_available()
        print(f"Target Competition: {DEFAULT_COMPETITION_NAME}")
        print(f"Model Slot: {DEFAULT_MODEL_NAME}")
        print(f"Credentials Configured: {creds.get('valid')}")
        print(f"crunch-cli Installed: {cli_ok}")

if __name__ == "__main__":
    main()
