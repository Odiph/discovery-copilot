"""One session as a readable document: what was said, and what the coach said about it.

`python -m app render` produces the transcript a human reads. It does not show
the COACH -- so the only way to review a suggestion against the moment that
produced it has been to read `transcript.jsonl` by eye, or to scroll a
rehearsal's console output with the log noise still in it.

This interleaves the two: every turn, and under it every candidate the coach
produced at that trigger, with its move, its band, its flags and whether it
actually reached the panel. Suppressed candidates are shown too, and that is
the point -- a review that only shows what survived cannot tell you whether the
critic is throwing away the good ones.

Reads a session directory and writes Markdown. No model call, no network.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmark.console import use_utf8_console  # noqa: E402

FLAGS = (
    ("is_leading", "leading"),
    ("is_hypothetical_or_future_tense", "hypothetical"),
    ("is_pitch", "pitch"),
    ("is_opinion_seeking", "opinion-seeking"),
    ("is_confirmation_only", "confirmation-only"),
    ("is_generic", "generic"),
    ("is_off_topic", "off-topic"),
)

BAND_MARK = {"clean": "clean", "uncertain": "uncertain", "clear_violation": "VIOLATION"}


def read(path: Path):
    segments, suggestions, events = [], [], []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except ValueError:
                continue
            kind = record.get("kind")
            if kind == "segment":
                segments.append(record)
            elif kind == "suggestion":
                suggestions.append(record)
            elif kind == "event":
                events.append(record)
    segments.sort(key=lambda r: r["seq"])
    return segments, suggestions, events


def turns(segments):
    """Consecutive same-speaker runs, joined -- the unit a human reads. The
    engine's ~5-word fragments are a transport detail and rendering them as
    separate lines makes the conversation unreadable."""
    grouped = []
    for segment in segments:
        if grouped and grouped[-1]["speaker"] == segment["speaker"]:
            grouped[-1]["seqs"].append(segment["seq"])
            grouped[-1]["text"] += " " + (segment.get("text") or "")
        else:
            grouped.append({
                "speaker": segment["speaker"],
                "seqs": [segment["seq"]],
                "text": segment.get("text") or "",
            })
    return grouped


def main(argv: "Optional[Sequence[str]]" = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("session")
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)

    session_dir = Path(args.session)
    segments, suggestions, events = read(session_dir / "transcript.jsonl")
    by_trigger: "dict[int, list[dict]]" = {}
    for row in suggestions:
        by_trigger.setdefault(row.get("trigger_seq"), []).append(row)

    meta = {}
    meta_path = session_dir / "session.json"
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except ValueError:
            meta = {}

    out: "list[str]" = []
    out.append(f"# Interview review — {session_dir.name}")
    out.append("")
    if meta.get("synthetic"):
        out.append("> ⚠️ **SYNTHETIC.** The interviewee is model-generated. Nothing here was "
                   "said by a person, and this session must never be extracted into the "
                   "fact corpus.")
        out.append("")
    shown = sum(1 for r in suggestions if r.get("shown"))
    out.append(f"{len(segments)} segments · {len(suggestions)} suggestions produced · "
               f"{shown} shown · {len(suggestions) - shown} suppressed")
    out.append("")
    out.append("Coach lines are indented under the turn that triggered them. "
               "**Suppressed candidates are included** — a review that only shows what "
               "survived cannot tell you whether the critic is discarding the good ones.")
    out.append("")
    out.append("---")
    out.append("")

    for turn in turns(segments):
        who = "**INTERVIEWER**" if turn["speaker"] == "interviewer" else "**INTERVIEWEE**"
        low, high = turn["seqs"][0], turn["seqs"][-1]
        span = f"{low}" if low == high else f"{low}–{high}"
        out.append(f"{who} · seq {span}")
        out.append("")
        out.append(f"> {turn['text'].strip()}")
        out.append("")
        fired = [seq for seq in turn["seqs"] if seq in by_trigger]
        for seq in fired:
            for row in by_trigger[seq]:
                flags = [label for field, label in FLAGS if row.get(field)]
                band = BAND_MARK.get(row.get("band"), str(row.get("band")))
                mark = "SHOWN" if row.get("shown") else "suppressed"
                bits = [f"`{row.get('move')}`", band, mark]
                if flags:
                    bits.append("flags: " + ", ".join(flags))
                out.append(f"- **COACH** ({' · '.join(bits)}) — trigger seq {seq}")
                text = (row.get("text") or "").strip()
                out.append(f"  - {text if text else '_(declined — no question)_'}")
                if row.get("rationale"):
                    out.append(f"  - _critic:_ {row['rationale']}")
        if fired:
            out.append("")

    text = "\n".join(out) + "\n"
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"written: {args.out}", file=sys.stderr)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    use_utf8_console()
    raise SystemExit(main())
