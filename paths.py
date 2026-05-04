"""Loads paths.env into os.environ without overriding values already in the environment.

Import this module before reading any path-shaped env var. It's stdlib-only
(no python-dotenv dependency) and idempotent.

Real shell-exported env vars always win — paths.env only fills in the blanks.
"""

import os

_REPO_ROOT = os.path.abspath(os.path.dirname(__file__))
_ENV_FILE = os.path.join(_REPO_ROOT, 'paths.env')


def _parse(line):
    line = line.split('#', 1)[0].strip()
    if not line or '=' not in line:
        return None
    key, _, value = line.partition('=')
    key = key.strip()
    value = value.strip()
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        value = value[1:-1]
    return key, value


def load(env_file=_ENV_FILE):
    """Merge non-empty values from paths.env into os.environ (does not override existing keys)."""
    if not os.path.exists(env_file):
        return
    with open(env_file) as f:
        for raw in f:
            kv = _parse(raw)
            if kv is None:
                continue
            key, value = kv
            if value == '' or key in os.environ:
                continue
            os.environ[key] = value


# Auto-load on import.
load()


REPO_ROOT = _REPO_ROOT
