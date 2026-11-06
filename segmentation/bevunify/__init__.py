"""RoboGeo BEV segmentation harness: GaussianLSS and CVT on the RoboGeo vehicle labels.

Importing this package puts the vendored upstream packages first on ``sys.path`` and
checks that they resolve to the vendored copies, so an installed copy of GaussianLSS or
cross_view_transformer cannot silently replace them.
"""
import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

_VENDORED = {
    "GaussianLSS": ROOT / "third_party" / "GaussianLSS",
    "cross_view_transformer": ROOT / "third_party" / "cross_view_transformers",
}

for _name, _repo in _VENDORED.items():
    if str(_repo) not in sys.path:
        sys.path.insert(0, str(_repo))
    _module = importlib.import_module(_name)
    _found = Path(_module.__file__).resolve().parent
    if _found != (_repo / _name).resolve():
        raise ImportError(
            f"'{_name}' was imported from {_found}, not from the vendored copy in {_repo}. "
            f"Uninstall the other copy or remove it from PYTHONPATH.")
