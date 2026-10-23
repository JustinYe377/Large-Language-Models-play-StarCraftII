import os
import sys

# Make `sc2_rl_agent.starcraftenv_test...` importable regardless of where
# pytest is launched from.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
