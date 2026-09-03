"""User callout aliases: one vocabulary at the LLM boundary (feature B).

Zone names from the game files stay canonical inside every stored artifact.
User-chosen callouts live in ``data/mapcards/<map>/aliases.json`` and are
applied by a :class:`Renamer` at exactly the places a human or the model
looks: prompts, tool output, brief text, lint vocabulary. The model is never
shown both names - user name if defined, game name otherwise.
"""

import hashlib
import json
import re
from pathlib import Path

# Aliases must survive backticked rendering and the lint tokenizer.
_ALIAS_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 _-]*$")


def _aliases_path(data_root: Path, map_name: str) -> Path:
    return data_root / "mapcards" / map_name / "aliases.json"


def load_aliases(data_root: Path, map_name: str) -> dict[str, str]:
    path = _aliases_path(data_root, map_name)
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return {str(k): str(v) for k, v in raw.items() if isinstance(raw, dict) and v}


def save_aliases(
    data_root: Path, map_name: str, aliases: dict[str, str], valid_zones: set[str]
) -> dict[str, str]:
    """Validate and persist; empty/whitespace alias values remove the alias."""
    cleaned: dict[str, str] = {}
    for canonical, alias in aliases.items():
        if canonical not in valid_zones:
            raise ValueError(f"Unknown zone: {canonical!r}")
        alias = (alias or "").strip()
        if not alias or alias == canonical:
            continue
        if not _ALIAS_RE.match(alias):
            raise ValueError(f"Alias {alias!r} contains invalid characters")
        if alias in valid_zones and alias != canonical:
            raise ValueError(f"Alias {alias!r} collides with another zone's game name")
        cleaned[canonical] = alias
    seen: dict[str, str] = {}
    for canonical, alias in cleaned.items():
        if alias in seen:
            raise ValueError(f"Duplicate alias {alias!r} for {seen[alias]!r} and {canonical!r}")
        seen[alias] = canonical

    path = _aliases_path(data_root, map_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cleaned, indent=2, sort_keys=True), encoding="utf-8")
    return cleaned


def alias_fingerprint(aliases: dict[str, str]) -> str:
    """Stable content hash used to invalidate caches when callouts change."""
    return hashlib.sha1(json.dumps(aliases, sort_keys=True).encode()).hexdigest()[:12]


class Renamer:
    """Applies user callouts to outbound text; maps them back for SQL literals."""

    def __init__(self, aliases: dict[str, str]):
        self.aliases = {k: v for k, v in (aliases or {}).items() if v and v != k}
        # Word boundary OR a formation-count prefix ("3xMiddle"), which glues the
        # zone to a word character and would otherwise dodge the rename.
        self._pattern = (
            re.compile(
                r"(?:\b|(?<=\dx))("
                + "|".join(re.escape(k) for k in sorted(self.aliases, key=len, reverse=True))
                + r")\b"
            )
            if self.aliases
            else None
        )
        self._reverse = {v: k for k, v in self.aliases.items()}
        self._literal = (
            re.compile(
                "(['\"])("
                + "|".join(re.escape(a) for a in sorted(self._reverse, key=len, reverse=True))
                + ")\\1"
            )
            if self._reverse
            else None
        )

    def __bool__(self) -> bool:
        return bool(self.aliases)

    def resolve(self, zone: str) -> str:
        return self.aliases.get(zone, zone)

    def rename_text(self, text: str) -> str:
        """Canonical -> user callout, whole words only ('team_Middle' untouched)."""
        if not self._pattern or not text:
            return text
        return self._pattern.sub(lambda m: self.aliases[m.group(1)], text)

    def unalias_sql(self, sql: str) -> str:
        """User callout -> canonical inside quoted SQL literals (the lake speaks canonical)."""
        if not self._literal or not sql:
            return sql
        return self._literal.sub(
            lambda m: f"{m.group(1)}{self._reverse[m.group(2)]}{m.group(1)}", sql
        )


def load_renamer(data_root: Path, map_name: str) -> Renamer:
    return Renamer(load_aliases(data_root, map_name))
