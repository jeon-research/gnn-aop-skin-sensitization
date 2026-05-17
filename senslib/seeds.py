"""Seed manifest loaded from seeds.json at the repo root."""

from __future__ import annotations

import json
from pathlib import Path

_data = json.loads((Path(__file__).resolve().parent.parent / 'seeds.json').read_text())

PRIMARY = tuple(_data['primary'])
SENSITIVITY = tuple(_data['sensitivity'])
ALL = tuple(_data['all'])
