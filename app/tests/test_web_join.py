"""Task 2 done criteria for the join surface (D-05, 03-01-PLAN.md §P7): token
generation, LAN IP resolution, URL assembly, QR degradation, and port-in-use
behaviour. Every failure path here is driven for real -- a monkeypatched
raise, not a mock of the assertion itself."""

from __future__ import annotations

import os
import socket

import pytest

from app.session.store import SessionStore, create_session_dir
from app.web import join as join_mod
from app.web import server as web_server


@pytest.fixture(autouse=True)
def _never_open_a_real_gui_viewer(monkeypatch):
    """`create_join` writes a real QR PNG and opens it via `os.startfile` (a
    real OS-level side effect -- launching Photos.exe on this Windows
    machine). Tests must never do that: patch it out globally for every test
    in this module so QR PNG *generation* is still exercised for real, but
    nothing gets opened."""
    monkeypatch.setattr(os, "startfile", lambda path: None, raising=False)


def _store(tmp_path) -> SessionStore:
    store = SessionStore(create_session_dir(tmp_path))
    store.write_metadata({"engine": "fake"})
    return store


def _events(store: SessionStore) -> list[dict]:
    import json

    return [
        json.loads(line)
        for line in store.transcript_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


# --- generate_join_token -------------------------------------------------------------


def test_generate_join_token_is_distinct_url_safe_and_long_enough():
    token_a = join_mod.generate_join_token()
    token_b = join_mod.generate_join_token()
    assert token_a != token_b
    assert len(token_a) >= 20
    # URL-safe: only characters token_urlsafe ever produces.
    allowed = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_")
    assert set(token_a) <= allowed


def test_two_create_join_calls_in_one_process_yield_different_tokens(tmp_path):
    store = _store(tmp_path)
    first = join_mod.create_join(port=18765, session_dir=store.session_dir, store=store, emit_qr=False)
    second = join_mod.create_join(port=18765, session_dir=store.session_dir, store=store, emit_qr=False)
    assert first.token != second.token


# --- URL assembly ---------------------------------------------------------------------


def test_create_join_assembles_url_from_resolved_lan_ip(tmp_path, monkeypatch):
    monkeypatch.setattr(join_mod, "get_lan_ip", lambda: "192.168.1.42")
    store = _store(tmp_path)
    info = join_mod.create_join(port=18765, session_dir=store.session_dir, store=store, emit_qr=False)
    assert info.lan_ip == "192.168.1.42"
    assert info.join_url == f"http://192.168.1.42:18765/s/{info.token}"


# --- Degradation: LAN IP failure never aborts ------------------------------------------


def test_lan_ip_failure_degrades_to_text_url_and_records_event(tmp_path, monkeypatch):
    def _raise() -> str:
        raise OSError("no route to host")

    monkeypatch.setattr(join_mod, "get_lan_ip", _raise)
    store = _store(tmp_path)

    info = join_mod.create_join(port=18765, session_dir=store.session_dir, store=store, emit_qr=True)  # must not raise

    assert info.lan_ip is None
    assert info.join_url == f"http://127.0.0.1:18765/s/{info.token}"
    events = [e for e in _events(store) if e.get("event") == "join-degraded"]
    assert any(e["detail"]["when"] == "lan-ip" for e in events)
    assert all("error_type" in e["detail"] for e in events)


# --- Degradation: QR failure (both terminal and PNG) never aborts ---------------------


def test_qr_terminal_failure_degrades_and_records_event(tmp_path, monkeypatch):
    monkeypatch.setattr(join_mod, "get_lan_ip", lambda: "192.168.1.42")

    def _raise(join_url: str) -> None:
        raise RuntimeError("terminal too narrow")

    monkeypatch.setattr(join_mod, "_render_terminal_qr", _raise)
    store = _store(tmp_path)

    info = join_mod.create_join(port=18765, session_dir=store.session_dir, store=store, emit_qr=True)  # must not raise

    assert info.join_url.startswith("http://192.168.1.42:18765/s/")
    events = [e for e in _events(store) if e.get("event") == "join-degraded"]
    assert any(e["detail"]["when"] == "qr-terminal" for e in events)


def test_qr_png_failure_degrades_and_records_event_but_join_still_proceeds(tmp_path, monkeypatch):
    monkeypatch.setattr(join_mod, "get_lan_ip", lambda: "192.168.1.42")

    def _raise(join_url: str, session_dir):
        raise OSError("disk full")

    monkeypatch.setattr(join_mod, "_render_qr_png", _raise)
    store = _store(tmp_path)

    info = join_mod.create_join(port=18765, session_dir=store.session_dir, store=store, emit_qr=True)  # must not raise

    assert info.qr_png_path is None
    events = [e for e in _events(store) if e.get("event") == "join-degraded"]
    assert any(e["detail"]["when"] == "qr-png" for e in events)


# --- Port-in-use: a typed, actionable error, never a silent fallback ------------------


def test_check_port_available_raises_typed_error_when_port_is_taken():
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.bind(("127.0.0.1", 0))
    probe.listen(1)
    taken_port = probe.getsockname()[1]
    try:
        with pytest.raises(web_server.PortInUseError) as exc_info:
            web_server.check_port_available(taken_port, host="127.0.0.1")
        message = str(exc_info.value)
        assert str(taken_port) in message
        assert "APP_PORT" in message
    finally:
        probe.close()


def test_check_port_available_does_not_raise_for_a_free_port():
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.bind(("127.0.0.1", 0))
    free_port = probe.getsockname()[1]
    probe.close()  # release it immediately -- now free again
    web_server.check_port_available(free_port, host="127.0.0.1")  # must not raise


# --- the QR toggle (user decision, 2026-08-10) ------------------------------------------


def test_with_the_qr_off_neither_form_is_rendered(tmp_path, monkeypatch):
    # The point of the toggle: nothing is drawn and no PNG is written, but the
    # join surface is otherwise untouched.
    calls: "list[str]" = []
    monkeypatch.setattr(join_mod, "_render_terminal_qr", lambda url: calls.append("terminal"))
    monkeypatch.setattr(join_mod, "_render_qr_png", lambda url, d: calls.append("png"))
    store = _store(tmp_path)
    try:
        info = join_mod.create_join(
            port=18765, session_dir=store.session_dir, store=store, emit_qr=False
        )
    finally:
        store.close()

    assert calls == []
    assert info.qr_png_path is None
    # The half that must NOT change: a phone can still join by typing this.
    assert info.token and info.join_url.endswith(f"/s/{info.token}")
    assert not (store.session_dir / join_mod.QR_PNG_FILENAME).exists()


def test_with_the_qr_on_both_forms_are_rendered(tmp_path, monkeypatch):
    calls: "list[str]" = []
    monkeypatch.setattr(join_mod, "_render_terminal_qr", lambda url: calls.append("terminal"))
    monkeypatch.setattr(
        join_mod, "_render_qr_png", lambda url, d: calls.append("png") or (d / "join-qr.png")
    )
    store = _store(tmp_path)
    try:
        join_mod.create_join(
            port=18765, session_dir=store.session_dir, store=store, emit_qr=True
        )
    finally:
        store.close()
    assert calls == ["terminal", "png"]


def test_the_qr_is_off_by_default_in_config():
    from app import config as config_mod

    assert config_mod.DEFAULT_EMIT_QR is False
