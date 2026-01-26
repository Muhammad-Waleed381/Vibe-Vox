"""Dialogue detection helpers."""

from __future__ import annotations

import re


QUOTE_PATTERN = re.compile(r"[\"\u201c\u201d].+?[\"\u201c\u201d]")
SCRIPT_LINE_PATTERN = re.compile(r"^\s*[A-Z][A-Z0-9_ ]{1,24}:\s+", re.MULTILINE)
SPEAKER_DASH_PATTERN = re.compile(r"^\s*[A-Z][A-Za-z0-9_ ]{1,24}\s+-\s+", re.MULTILINE)


def has_dialogue(text: str) -> bool:
    """Return True if text contains dialogue-like markers."""
    if QUOTE_PATTERN.search(text):
        return True
    if SCRIPT_LINE_PATTERN.search(text):
        return True
    if SPEAKER_DASH_PATTERN.search(text):
        return True
    return False
