"""D-05: the per-run join surface -- an unguessable token, the LAN IP the
phone needs to reach this machine, the assembled join URL, and a QR code shown
two ways. This is the ONLY place a join URL is assembled (03-CONTEXT.md D-05,
03-RESEARCH.md §4).

Shaped like `app/capture/devices.py`: small, pure, injectable free functions
returning one frozen dataclass, rather than a class. Every failure in here
DEGRADES rather than aborts a session -- following `app/session/audio_writer.py`'s
`_disable()` pattern: catch, log `error_type` only (never the token or the
URL -- T-03-04), record a named store event, and continue with a text-only
join URL. An interview that starts without a pretty QR is fine; an interview
that does not start is not.

The per-run token is NOT the OPS-03 "one secret" -- it is generated fresh in
memory every run (03-RESEARCH.md §4) and is never written anywhere on its
own. It DOES end up on disk as part of the join URL encoded into
`QR_PNG_FILENAME` inside the session directory (a deliberate QR-reliability
feature, not an oversight -- see `_render_qr_png()` below): `sessions/` is
gitignored and the PNG dies with the run alongside everything else there, so
the practical exposure is the same as any other file under a run's own
session directory, never a persisted secret store."""

from __future__ import annotations

import logging
import os
import secrets
import socket
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from app.session.store import SessionStore

logger = logging.getLogger(__name__)

# 128 bits of entropy, ~22 URL-safe characters -- small enough for a QR/URL,
# large enough that guessing it within an interview's lifetime is infeasible.
# Never `random`: its own documentation rules it out for anything
# security-relevant (03-RESEARCH.md §4).
TOKEN_NBYTES = 16

# The UDP-connect/getsockname() trick: ask the OS routing table which
# interface it would use to reach the outside world, rather than enumerating
# adapters and re-implementing "which one is real" (03-RESEARCH.md §4). No
# packet is actually sent -- UDP connect() only resolves a route.
_LAN_PROBE_HOST = "8.8.8.8"
_LAN_PROBE_PORT = 80

QR_PNG_FILENAME = "join-qr.png"


@dataclass(frozen=True)
class JoinInfo:
    """Everything a caller needs to print/show the join surface. `lan_ip` and
    `qr_png_path` are `None` when their resolution degraded -- callers must
    handle both, never assume they succeeded."""

    token: str
    lan_ip: Optional[str]
    port: int
    join_url: str
    qr_png_path: Optional[Path]


def generate_join_token() -> str:
    """A fresh, cryptographically-random token. Distinct on every call --
    the property that keeps a token seen once dead the moment this run ends
    (T-03-02)."""
    return secrets.token_urlsafe(TOKEN_NBYTES)


def get_lan_ip() -> str:
    """Resolves this machine's LAN-facing IPv4 address via the UDP-connect
    trick. Raises `OSError` on failure -- callers degrade, never propagate
    this to a crashed session. Known failure mode: an active VPN or virtual
    adapter can make this return THAT adapter's address instead of the real
    Wi-Fi adapter's -- callers must print an explicit sanity line asking the
    user to confirm (03-RESEARCH.md §4)."""
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.connect((_LAN_PROBE_HOST, _LAN_PROBE_PORT))
        return probe.getsockname()[0]
    finally:
        probe.close()


def _render_terminal_qr(join_url: str) -> None:
    import qrcode

    qr = qrcode.QRCode()
    qr.add_data(join_url)
    qr.make()
    qr.print_ascii(invert=True)


def _render_qr_png(join_url: str, session_dir: Path) -> Path:
    import qrcode

    qr = qrcode.QRCode()
    qr.add_data(join_url)
    qr.make()
    image = qr.make_image()
    png_path = session_dir / QR_PNG_FILENAME
    image.save(png_path)
    if hasattr(os, "startfile"):
        os.startfile(png_path)  # Windows-only; simply absent elsewhere
    return png_path


def _emit_both_qr_forms(
    join_url: str, session_dir: Path, store: SessionStore
) -> "Optional[Path]":
    """Renders the terminal QR and the PNG, degrading independently.

    Each form is tried separately and neither failure aborts the session --
    an interview that starts without a pretty QR is fine; an interview that
    does not start is not. Returns the PNG path, or `None` if that render
    failed. Called only when the QR is enabled."""
    qr_png_path: "Optional[Path]" = None
    try:
        _render_terminal_qr(join_url)
    except Exception as exc:  # noqa: BLE001 -- degrade, never abort a session over a QR failure
        logger.warning("terminal qr rendering failed error_type=%s", type(exc).__name__)
        store.append_event(
            event="join-degraded",
            detail={"when": "qr-terminal", "error_type": type(exc).__name__},
        )
    try:
        qr_png_path = _render_qr_png(join_url, session_dir)
    except Exception as exc:  # noqa: BLE001 -- same degrade-not-abort contract
        logger.warning("qr png rendering failed error_type=%s", type(exc).__name__)
        store.append_event(
            event="join-degraded",
            detail={"when": "qr-png", "error_type": type(exc).__name__},
        )
    return qr_png_path


def create_join(
    *, port: int, session_dir: Path, store: SessionStore, emit_qr: bool
) -> JoinInfo:
    """Assembles the full join surface for one run: token, LAN IP, join URL,
    and -- when `emit_qr` -- both QR forms (terminal ANSI + PNG). Never raises
    -- a QR or LAN-IP failure degrades to a text-only join URL, records a named
    store event, and the session proceeds (03-RESEARCH.md §4, PATTERNS.md
    `audio_writer._disable()` precedent).

    `emit_qr` has NO default. The QR is off by default at the config layer
    (`DEFAULT_EMIT_QR`), and a default here as well would mean two places
    decide, with the quieter one winning silently -- the same forcing-function
    reasoning as `load_facts`'s explicitly-nullable `sessions_root`.

    With it off, NOTHING about the join surface changes except the two QR
    renders: the token is still fresh, the LAN IP still resolved, and the join
    URL still printed, so a phone can still join by typing it."""
    token = generate_join_token()

    lan_ip: Optional[str] = None
    try:
        lan_ip = get_lan_ip()
    except OSError as exc:
        logger.warning("lan ip resolution failed error_type=%s", type(exc).__name__)
        store.append_event(
            event="join-degraded",
            detail={"when": "lan-ip", "error_type": type(exc).__name__},
        )

    host_for_url = lan_ip if lan_ip is not None else "127.0.0.1"
    join_url = f"http://{host_for_url}:{port}/s/{token}"

    qr_png_path: Optional[Path] = None
    if emit_qr:
        qr_png_path = _emit_both_qr_forms(join_url, session_dir, store)

    print(f"Join URL: {join_url}")
    if lan_ip is not None:
        print(
            f"Resolved LAN IP: {lan_ip} -- confirm this matches your Wi-Fi adapter's "
            f"address (check ipconfig if unsure), not a VPN or virtual adapter."
        )
    else:
        print(
            "Could not resolve a LAN IP automatically -- share the join URL above "
            "manually (check ipconfig for your Wi-Fi adapter's IPv4 address)."
        )

    return JoinInfo(token=token, lan_ip=lan_ip, port=port, join_url=join_url, qr_png_path=qr_png_path)
