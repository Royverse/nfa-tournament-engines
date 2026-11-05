"""NumerAPI Client wrapper with credential discovery, retries, and error reporting."""
import os
import sys
import time
from typing import Dict, Any, Optional, Tuple
try:
    from numerapi import NumerAPI
except ImportError:
    NumerAPI = Any  # type: ignore
from .config import TOURNAMENT_ID, DEFAULT_MODEL_NAME

def get_auth_credentials() -> Tuple[Optional[str], Optional[str]]:
    """Extract public_id and secret_key from environment variables."""
    pub = os.getenv("NUMERAI_PUBLIC_ID")
    sec = os.getenv("NUMERAI_SECRET_KEY")
    if pub and sec:
        return pub.strip(), sec.strip()

    mcp_auth = os.getenv("NUMERAI_MCP_AUTH")
    if mcp_auth:
        clean = mcp_auth.replace("Token ", "").strip()
        parts = clean.split("$")
        if len(parts) == 2:
            return parts[0].strip(), parts[1].strip()

    return None, None

def create_client(verbosity: str = "INFO") -> Any:
    """Create a configured NumerAPI instance."""
    if NumerAPI is Any:
        raise ImportError("numerapi is required to create a NumerAPI client.")
    pub, sec = get_auth_credentials()
    return NumerAPI(public_id=pub, secret_key=sec, verbosity=verbosity)

def verify_credentials(napi: NumerAPI) -> Dict[str, Any]:
    """Verify that credentials are valid and can access model accounts."""
    try:
        models = napi.get_models()
        return {"valid": True, "models": models}
    except Exception as e:
        return {"valid": False, "error": str(e)}

def get_current_round(napi: NumerAPI, tournament: int = TOURNAMENT_ID) -> Optional[int]:
    """Get the current open tournament round number."""
    try:
        return napi.get_current_round(tournament=tournament)
    except Exception as e:
        print(f"[Client] Warning: Failed to get current round: {e}", file=sys.stderr)
        return None

def check_submission_status(
    napi: NumerAPI,
    round_number: int,
    model_name: str = DEFAULT_MODEL_NAME,
) -> Tuple[bool, Optional[str]]:
    """Check if model has already submitted for the given round.
    
    Returns (has_submitted, submission_id_or_filename).
    """
    try:
        query = """
        query Account {
            account {
                models {
                    name
                    submissions {
                        id
                        filename
                        round {
                            number
                        }
                    }
                }
            }
        }
        """
        res = napi.raw_query(query, authorization=True)
        models = res.get("data", {}).get("account", {}).get("models", [])
        for m in models:
            if m.get("name", "").lower() == model_name.lower():
                for sub in m.get("submissions", []):
                    if sub.get("round", {}).get("number") == round_number:
                        return True, sub.get("id") or sub.get("filename")
    except Exception as e:
        print(f"[Client] Warning: Could not check submission status: {e}", file=sys.stderr)
    return False, None

def download_dataset_safe(
    napi: NumerAPI,
    remote_path: str,
    dest_path: str,
    max_retries: int = 3,
) -> bool:
    """Download a dataset file with retry logic."""
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
        print(f"[Client] Using cached file: {dest_path}")
        return True

    for attempt in range(1, max_retries + 1):
        try:
            print(f"[Client] Downloading {remote_path} (attempt {attempt}/{max_retries})...")
            napi.download_dataset(remote_path, dest_path=dest_path)
            if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
                print(f"[Client] Successfully downloaded {dest_path}")
                return True
        except Exception as e:
            print(f"[Client] Download failed: {e}", file=sys.stderr)
            if attempt < max_retries:
                time.sleep(attempt * 2)

    return False
