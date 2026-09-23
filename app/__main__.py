"""Entry point for `python -m app`. Not in the plan's file list -- added under
deviation Rule 3 (blocking issue): `python -m app start ...` cannot run at all
without this module, and the plan's own `<verify>`/human-check steps invoke
exactly that command."""

from __future__ import annotations

import sys

from app.main import main

if __name__ == "__main__":
    sys.exit(main())
