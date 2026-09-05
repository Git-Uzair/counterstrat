"""Analysis scope identity: one hash per (team, map, exact match set).

The hash is the shared key for a scope's chat session AND its First Read
cache, so re-selecting the same matches resurfaces the same conversation and
read, while adding or removing a match yields a clean slate. Hashing the
concrete match ids (never an "all" sentinel) means a changed corpus is a new
scope by construction.
"""

import hashlib


def scope_hash(team_id: str, map_name: str, match_ids: list[str]) -> str:
    key = f"{team_id}|{map_name}|{','.join(sorted(match_ids))}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]
