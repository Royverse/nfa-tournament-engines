"""CrunchDAO Client and CLI wrapper with credential discovery, retries, and runner execution."""
import os
import sys
import subprocess
import shutil
from typing import Dict, Any, Optional, Tuple, List
from .config import DEFAULT_COMPETITION_NAME, DEFAULT_MODEL_NAME, BASE_DIR

def get_auth_token() -> Optional[str]:
    """Extract authentication token from environment variables."""
    token = os.getenv("CRUNCHDAO_TOKEN") or os.getenv("CRUNCH_TOKEN") or os.getenv("CRUNCHDAO_API_KEY")
    if token:
        clean = token.strip()
        if clean.startswith("Bearer "):
            clean = clean.replace("Bearer ", "").strip()
        return clean
    return None

def verify_credentials() -> Dict[str, Any]:
    """Verify that CrunchDAO token credentials are configured."""
    token = get_auth_token()
    if not token:
        return {
            "valid": False,
            "error": "Missing CRUNCHDAO_TOKEN or CRUNCH_TOKEN in environment",
        }
    
    # Check minimum token sanity
    if len(token) < 10:
        return {
            "valid": False,
            "error": "CRUNCHDAO_TOKEN appears too short or malformed",
        }

    # No preview of the token: a slice of a secret is not masked in Actions logs.
    return {"valid": True}

def check_cli_available() -> bool:
    """Check if crunch-cli is installed and accessible."""
    return shutil.which("crunch") is not None

def run_crunch_command(args: List[str], cwd: Optional[str] = None) -> Tuple[int, str, str]:
    """Execute a crunch-cli command safely within the CrunchDAO project directory."""
    work_dir = cwd or BASE_DIR
    cmd = ["crunch"] + args
    try:
        proc = subprocess.run(
            cmd,
            cwd=work_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=300,  # 5 minutes timeout
        )
        return proc.returncode, proc.stdout, proc.stderr
    except FileNotFoundError:
        return 127, "", "crunch-cli binary not found on PATH. Run: pip install crunch-cli"
    except subprocess.TimeoutExpired:
        return 124, "", "crunch-cli command timed out after 300s"
    except Exception as e:
        return 1, "", str(e)

def setup_competition_workspace(
    competition: str = DEFAULT_COMPETITION_NAME,
    model_name: str = DEFAULT_MODEL_NAME,
    token: Optional[str] = None,
) -> Dict[str, Any]:
    """Run `crunch setup` to configure local workspace against CrunchDAO."""
    auth_token = token or get_auth_token()
    if not auth_token:
        return {"success": False, "error": "No CrunchDAO token provided"}

    args = [
        "setup",
        competition,
        model_name,
        "--token",
        auth_token,
    ]
    code, stdout, stderr = run_crunch_command(args)
    if code == 0 or "already setup" in (stdout + stderr).lower():
        return {"success": True, "output": stdout}
    return {"success": False, "code": code, "stdout": stdout, "stderr": stderr}

def test_local_submission() -> Dict[str, Any]:
    """Run local `crunch test` to validate code conforms to CrunchDAO benchmark specifications."""
    code, stdout, stderr = run_crunch_command(["test"])
    return {
        "success": code == 0,
        "code": code,
        "stdout": stdout,
        "stderr": stderr,
    }

def push_submission(message: str = "Autonomous submission from NFA Engine") -> Dict[str, Any]:
    """Push model code and resources to CrunchDAO Hub."""
    args = ["push", "-m", message]
    code, stdout, stderr = run_crunch_command(args)
    return {
        "success": code == 0,
        "code": code,
        "stdout": stdout,
        "stderr": stderr,
    }
