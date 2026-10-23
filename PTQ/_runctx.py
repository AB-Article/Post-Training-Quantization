"""Make every analysis script runnable from any working directory.

All scripts in this folder address their inputs and outputs relative to
``analysis/`` (``results/...``, ``im100_acc/...``, ``../figures/...``).  Before
this module existed, invoking one from anywhere else -- ``python3
analysis/guideline.py`` from the repository root, for instance -- failed with a
bare ``FileNotFoundError``.

Importing this module first switches the process to ``analysis/``.  Relative
paths given on the command line are resolved against the *caller's* directory
beforehand, so ``python3 analysis/predictor.py --data-root ./nn-dataset`` keeps
pointing at the checkout the user meant.  Importing it when the working
directory is already ``analysis/`` does nothing.
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))


def _absolutize(arg: str) -> str:
    """Resolve an existing relative path argument against the current CWD."""
    if arg.startswith('--') and '=' in arg:
        flag, _, value = arg.partition('=')
        return f'{flag}={_absolutize(value)}' if value else arg
    if not arg or arg.startswith('-'):
        return arg
    return os.path.abspath(arg) if os.path.exists(arg) else arg


def activate() -> None:
    if os.path.abspath(os.getcwd()) == _HERE:
        return
    sys.argv[1:] = [_absolutize(a) for a in sys.argv[1:]]
    os.chdir(_HERE)


activate()
