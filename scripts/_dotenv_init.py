"""
Helper: load .env BEFORE Python finishes initializing subprocess-spawning libs.

Import this FIRST in any script that uses F5-TTS or other libs that spawn
Python subprocesses via multiprocessing. If PYTHONHASHSEED is set to empty
string in the parent shell, the spawned subprocess crashes during preinit
with: "PYTHONHASHSEED must be 'random' or an integer in range [0; 4294967295]"

Usage in scripts:
    # At very top, before any other imports
    from _dotenv_init import init_env_then_reexec
    init_env_then_reexec(__file__)
"""

import os
import sys
from pathlib import Path


def _load_dotenv_silently(path: Path) -> dict[str, str]:
    """Minimal .env parser — no python-dotenv dep required for early init."""
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip().strip("'").strip('"')
    return out


def init_env_then_reexec(script_file: str) -> None:
    """Load .env, fix PYTHONHASHSEED, and re-exec self if it changed.

    Called at the top of any script that loads F5-TTS or anything else
    that internally uses multiprocessing.
    """
    project_root = Path(script_file).parent.parent
    env_file = project_root / ".env"
    env = _load_dotenv_silently(env_file)

    # The critical fix: empty PYTHONHASHSEED crashes spawned Python subprocesses.
    # Either set it to a valid value, or remove it entirely.
    current = os.environ.get("PYTHONHASHSEED")
    desired = env.get("PYTHONHASHSEED", "random")

    needs_reexec = False
    if current is None or current == "":
        # Apply our valid value
        os.environ["PYTHONHASHSEED"] = desired
        needs_reexec = True
    elif current != desired and desired != "random":
        # Caller explicitly set something different in .env; honor it
        os.environ["PYTHONHASHSEED"] = desired
        needs_reexec = True

    # Also apply any other .env vars (CLIENT_ID etc) — but DON'T re-exec for them
    for k, v in env.items():
        if k != "PYTHONHASHSEED":
            os.environ.setdefault(k, v)

    if needs_reexec:
        # Re-exec self so that Python's own hash-seed init reads the new value
        os.execv(sys.executable, [sys.executable, script_file] + sys.argv[1:])


def kill_stale_python(extra_patterns: list[str] | None = None) -> int:
    """Kill stale F5-TTS / eval / Whisper Python procs to free MPS memory.

    Each main() call should invoke this at the top — orphans from previous
    runs (even "successful" ones) accumulate and OOM the next launch on the
    18 GB M3 Pro. Excludes the current PID so it's safe to call from inside
    the very process you don't want to kill.

    Returns the number of processes killed.
    """
    import re
    import signal
    import subprocess
    import time

    patterns = [
        r"scripts/finetune_f5\.py",
        r"scripts/posthoc_eval\.py",
        r"scripts/f5_infer\.py",
    ]
    if extra_patterns:
        patterns.extend(extra_patterns)
    combined = "|".join(patterns)
    my_pid = os.getpid()

    try:
        r = subprocess.run(["pgrep", "-fl", combined],
                           capture_output=True, text=True, timeout=5)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return 0
    if r.returncode != 0:
        return 0  # no matches

    killed = 0
    for line in r.stdout.splitlines():
        parts = line.strip().split(None, 1)
        if not parts:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        if pid == my_pid:
            continue
        try:
            os.kill(pid, signal.SIGTERM)
            killed += 1
            print(f"  [kill_stale_python] SIGTERM → PID {pid}: {parts[1][:80] if len(parts) > 1 else ''}",
                  file=sys.stderr)
        except (ProcessLookupError, PermissionError):
            pass
    if killed:
        time.sleep(2)  # let MPS pool reclaim memory
    return killed
