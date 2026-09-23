"""Entry point for `python -m app.mcp`.

Its own entry point rather than a subcommand of `app/main.py`, deliberately:
another agent holds that module, and more importantly this process owns
stdout absolutely -- a single `print` from anything `main.py` imports would
corrupt the protocol stream. A CLI subcommand wiring this in is a later,
separate commit.

**This module is the ONE injection point**, the same role `app/main.py` plays
for every other surface: nothing under `app/mcp/` reads config or resolves a
root internally -- `repo_root` is threaded through `serve()` as a parameter,
exactly as every loader under `app/coach/` and `app/session/` takes its
directory as one.

Logging goes to STDERR, always. stdout is the transport.
"""

from __future__ import annotations

import logging
import sys

from app.mcp.server import serve


def main() -> int:
    logging.basicConfig(stream=sys.stderr, level=logging.WARNING)
    # `repo_root=None` lets `app/projects.py` anchor to the repository root
    # it computes from its own file location -- never the working directory,
    # which for an MCP server is whatever the client happened to launch it
    # from.
    serve(sys.stdin.buffer, sys.stdout.buffer, repo_root=None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
