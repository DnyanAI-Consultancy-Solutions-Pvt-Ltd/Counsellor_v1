from __future__ import annotations

from collections import Counter
from models.schemas import Recommendation, FeedbackChangeSet


def deduplicate(rows: list[Recommendation]) -> list[Recommendation]:
    unique: dict[str, Recommendation] = {}
    order: list[str] = []
    for row in rows:
        key = row.row_key
        if key not in unique:
            order.append(key)
        unique[key] = row
    return [unique[key] for key in order]


def apply_change_set(existing: list[Recommendation], changes: FeedbackChangeSet) -> list[Recommendation]:
    removed = {key.strip().lower() for key in changes.remove_keys}
    rows = [row for row in existing if row.row_key not in removed]
    by_key = {row.row_key: row for row in rows}
    order = [row.row_key for row in rows]

    for row in changes.add_or_update:
        key = row.row_key
        if key not in by_key:
            order.append(key)
        by_key[key] = row

    rows = [by_key[key] for key in order if key in by_key]

    if changes.sort_mode == "cutoff_desc":
        rows.sort(key=lambda row: row.cutoff_percentile if row.cutoff_percentile is not None else -1, reverse=True)
    elif changes.sort_mode == "cutoff_asc":
        rows.sort(key=lambda row: row.cutoff_percentile if row.cutoff_percentile is not None else 101)
    elif changes.sort_mode == "preferred_order" and changes.preferred_order:
        rank = {key.strip().lower(): index for index, key in enumerate(changes.preferred_order)}
        rows.sort(key=lambda row: rank.get(row.row_key, len(rank) + order.index(row.row_key)))

    return deduplicate(rows)


def unique_college_count(rows: list[Recommendation]) -> int:
    return len({row.college.strip().lower() for row in rows if row.college.strip()})


def counts(rows: list[Recommendation]) -> dict[str, int]:
    zone_colleges: dict[str, set[str]] = {"Dream": set(), "Target": set(), "Safe": set()}
    for row in rows:
        zone_colleges.setdefault(row.match_category, set()).add(row.college.strip().lower())
    return {
        "Dream": len(zone_colleges.get("Dream", set())),
        "Target": len(zone_colleges.get("Target", set())),
        "Safe": len(zone_colleges.get("Safe", set())),
        "Total": unique_college_count(rows),
        "Rows": len(rows),
    }
