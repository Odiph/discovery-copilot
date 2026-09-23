"""WCAG contrast, computed from the shipped CSS rather than asserted about it.

Both surfaces failed here, and neither failure is visible to a reader of the
diff that introduced it: a hex is just a hex, and the number that matters is a
ratio against a background declared somewhere else in the file. So this reads
the real stylesheets, resolves each foreground against the surface it is
actually painted on, and computes the ratio.

The completeness assertions are the load-bearing half. A test pinning only the
values known to be wrong today passes forever while the next colour someone
adds goes unchecked -- which is exactly how three of these shipped. Adding a
colour or a surface to either stylesheet fails this file until the pair is
listed and its ratio computed.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

STATIC = Path(__file__).resolve().parent.parent / "static"
PHONE = STATIC / "style.css"
HUB = STATIC / "review" / "review.css"

AA_NORMAL = 4.5


def _channel(component: int) -> float:
    c = component / 255.0
    return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4


def luminance(hex_colour: str) -> float:
    h = hex_colour.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return 0.2126 * _channel(r) + 0.7152 * _channel(g) + 0.0722 * _channel(b)


def contrast(foreground: str, background: str) -> float:
    a, b = luminance(foreground), luminance(background)
    lighter, darker = max(a, b), min(a, b)
    return (lighter + 0.05) / (darker + 0.05)


def test_the_contrast_maths_matches_a_known_reference():
    """A ratio function that is wrong in the same direction everywhere would
    pass every assertion below while approving unreadable text."""
    assert contrast("#ffffff", "#000000") == pytest.approx(21.0, abs=0.01)
    assert contrast("#777777", "#ffffff") == pytest.approx(4.48, abs=0.02)
    # Order must not matter.
    assert contrast("#12151c", "#7d8595") == pytest.approx(contrast("#7d8595", "#12151c"))


def hexes(path: Path, prop: str) -> "set[str]":
    return {m.lower() for m in re.findall(rf"{prop}:\s*(#[0-9a-fA-F]{{6}})", path.read_text(encoding="utf-8"))}


# --- the phone -------------------------------------------------------------------
# Dark-only surface. Three backgrounds carry body text, plus two status tints
# that bring their own foregrounds.
PHONE_SURFACES = {"#0b0d12", "#171c26", "#12151c", "#2a2410", "#3a1414"}
PHONE_PAIRS = [
    # foreground,  background,          what it is
    ("#eaeaea", "#0b0d12", "body text"),
    ("#eaeaea", "#171c26", "the next question itself"),
    ("#8b93a3", "#171c26", "next-question label"),
    ("#7d8595", "#171c26", "next-question resting state"),
    ("#7d8595", "#12151c", "suggestion label and resting state"),
    ("#9099a8", "#12151c", "alternative suggestion"),
    ("#7d8595", "#0b0d12", "transcript timestamps"),
    ("#b8965a", "#2a2410", "gap marker"),
    ("#f0a0a0", "#3a1414", "server status"),
]


@pytest.mark.parametrize("fg,bg,what", PHONE_PAIRS)
def test_phone_text_clears_aa(fg, bg, what):
    ratio = contrast(fg, bg)
    assert ratio >= AA_NORMAL, f"{what}: {fg} on {bg} is {ratio:.2f}:1, needs {AA_NORMAL}"


def test_phone_dim_token_is_sized_against_its_lightest_surface():
    """#7d8595 sits on three surfaces. Passing on the darkest proves nothing --
    the binding one is the lightest, #171c26, and it is the tightest of the
    three at just over AA."""
    ratios = {bg: contrast("#7d8595", bg) for bg in ("#0b0d12", "#12151c", "#171c26")}
    assert min(ratios, key=ratios.get) == "#171c26"
    assert ratios["#171c26"] >= AA_NORMAL


def composite(foreground: str, background: str, alpha: float) -> str:
    """What a translucent foreground actually renders as. CSS `opacity` blends
    in the element's own colour space, so this is a plain sRGB mix."""
    f, b = foreground.lstrip("#"), background.lstrip("#")
    parts = [
        round(alpha * int(f[i:i + 2], 16) + (1 - alpha) * int(b[i:i + 2], 16))
        for i in (0, 2, 4)
    ]
    return "#" + "".join(f"{p:02x}" for p in parts)


def test_the_grounding_banner_still_clears_aa_through_its_opacity():
    """`.grounding-region` carries `opacity: 0.55` and `.grounding-text` sets no
    colour of its own, so it inherits the body's #eaeaea and renders as a blend.
    A ratio computed on the declared colour would report 15:1 and be wrong about
    what anyone sees; the real figure is nearer 5.4:1, which passes -- but it is
    the one value here that a small opacity change could quietly sink."""
    rendered = composite("#eaeaea", "#0b0d12", 0.55)
    assert contrast(rendered, "#0b0d12") >= AA_NORMAL
    assert "opacity: 0.55;" in PHONE.read_text(encoding="utf-8"), (
        "the opacity this test assumes has moved -- recompute the blend"
    )


def test_no_unreviewed_colour_or_surface_appeared_in_the_phone_css():
    known_fg = {fg for fg, _, _ in PHONE_PAIRS}
    assert hexes(PHONE, "background") <= PHONE_SURFACES, "new surface -- add its pairs above"
    assert hexes(PHONE, "color") <= known_fg, "new text colour -- add its pair above"


# --- the hub ---------------------------------------------------------------------
# `color-scheme: light dark` with no background of its own, so it renders on the
# UA Canvas. Chrome's dark canvas is #121212; white is the light one.
CANVAS_LIGHT = "#ffffff"
CANVAS_DARK = "#121212"


def _hub_tokens() -> "tuple[dict[str, str], dict[str, str]]":
    """Parsed out of the stylesheet, never restated here.

    A table of expected values copied into the test would pass unchanged if
    someone reverted the CSS -- the assertion would be comparing the test to
    itself. Reading the file is what makes a revert fail."""
    text = HUB.read_text(encoding="utf-8")
    # Tolerant of the block being absent so that deleting it fails as a named
    # assertion below rather than as a collection error here -- a stack trace
    # at import time says "the test is broken", which is the wrong diagnosis.
    marker = text.find("prefers-color-scheme: dark")
    split = len(text) if marker < 0 else marker
    pattern = r"(--[a-z]+):\s*(#[0-9a-fA-F]{6})"
    light = {t: c.lower() for t, c in re.findall(pattern, text[:split])}
    dark = dict(light)
    dark.update({t: c.lower() for t, c in re.findall(pattern, text[split:])})
    return light, dark


HUB_LIGHT, HUB_DARK = _hub_tokens()


@pytest.mark.parametrize("token,colour", sorted(HUB_LIGHT.items()))
def test_hub_light_mode_clears_aa(token, colour):
    ratio = contrast(colour, CANVAS_LIGHT)
    assert ratio >= AA_NORMAL, f"{token} {colour} on white is {ratio:.2f}:1"


@pytest.mark.parametrize("token,colour", sorted(HUB_DARK.items()))
def test_hub_dark_mode_clears_aa(token, colour):
    ratio = contrast(colour, CANVAS_DARK)
    assert ratio >= AA_NORMAL, f"{token} {colour} on {CANVAS_DARK} is {ratio:.2f}:1"


def test_one_token_set_could_not_have_served_both_canvases():
    """Why the dark block exists at all, asserted rather than left in a comment:
    AA against white caps luminance below what AA against #121212 requires, so
    the ranges do not overlap and no single value is correctable."""
    highest_that_passes_on_white = 1.05 / AA_NORMAL - 0.05
    lowest_that_passes_on_dark = AA_NORMAL * (luminance(CANVAS_DARK) + 0.05) - 0.05
    assert highest_that_passes_on_white < lowest_that_passes_on_dark


def test_every_semantic_token_has_its_own_dark_value():
    """A fourth token added to `:root` and not to the dark block inherits the
    light value and reintroduces exactly the bug this file was written for --
    silently, since it renders fine in the light mode most people develop in."""
    text = HUB.read_text(encoding="utf-8")
    marker = text.find("prefers-color-scheme: dark")
    assert marker >= 0, "the dark-mode block is gone; every token reverts to its light value"
    overridden = set(re.findall(r"(--[a-z]+):\s*#", text[marker:]))
    assert set(HUB_LIGHT) == overridden, (
        f"tokens with no dark override: {set(HUB_LIGHT) - overridden}"
    )
    assert HUB_LIGHT != HUB_DARK, "the dark block sets nothing the light one did not"
