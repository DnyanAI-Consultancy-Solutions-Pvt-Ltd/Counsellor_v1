from __future__ import annotations

import re
from collections import Counter
from copy import deepcopy
from typing import Any, Iterable


class PreferenceService:
    """Deterministic list operations and post-action validation."""

    ZONES = ("Dream", "Target", "Safe")

    @staticmethod
    def key(item: dict[str, Any]) -> tuple[str, str]:
        return (
            str(item.get("college", "")).strip().casefold(),
            str(item.get("branch", "")).strip().casefold(),
        )

    def normalise(self, recommendations: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for raw in recommendations:
            if not isinstance(raw, dict):
                continue
            item = deepcopy(raw)
            key = self.key(item)
            if not all(key) or key in seen:
                continue
            seen.add(key)
            output.append(item)
        return self.resequence(output)

    @staticmethod
    def resequence(recommendations: list[dict[str, Any]]) -> list[dict[str, Any]]:
        for index, item in enumerate(recommendations, start=1):
            item["rank"] = index
        return recommendations

    def remove_matching(
        self,
        recommendations: list[dict[str, Any]],
        *,
        field: str | None,
        value: str,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        needle = self._normalise_text(value)
        kept: list[dict[str, Any]] = []
        removed: list[dict[str, Any]] = []

        for item in recommendations:
            haystack = self._search_text(item, field)
            if needle and needle in haystack:
                removed.append(item)
            else:
                kept.append(item)

        return self.resequence(kept), removed

    def add_exact(
        self,
        current: list[dict[str, Any]],
        candidates: Iterable[dict[str, Any]],
        count: int,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        if count <= 0:
            return self.resequence(current), []

        output = self.normalise(current)
        seen = {self.key(item) for item in output}
        added: list[dict[str, Any]] = []

        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            key = self.key(candidate)
            if not all(key) or key in seen:
                continue
            output.append(deepcopy(candidate))
            added.append(deepcopy(candidate))
            seen.add(key)
            if len(added) == count:
                break

        return self.resequence(output), added

    def rerank(
        self,
        recommendations: list[dict[str, Any]],
        *,
        field: str | None,
        value: str,
        direction: str = "top",
    ) -> list[dict[str, Any]]:
        needle = self._normalise_text(value)

        def matches(item: dict[str, Any]) -> bool:
            return bool(needle and needle in self._search_text(item, field))

        matched = [item for item in recommendations if matches(item)]
        unmatched = [item for item in recommendations if not matches(item)]
        ordered = matched + unmatched if direction.casefold() != "bottom" else unmatched + matched
        return self.resequence(ordered)

    def validate(
        self,
        recommendations: list[dict[str, Any]],
        requested_counts: dict[str, int] | None = None,
    ) -> dict[str, Any]:
        duplicate_count = len(recommendations) - len({self.key(x) for x in recommendations})
        zone_counts = Counter(str(item.get("zone", "")) for item in recommendations)
        missing_by_zone: dict[str, int] = {}
        if requested_counts:
            missing_by_zone = {
                zone: max(0, int(required) - int(zone_counts.get(zone, 0)))
                for zone, required in requested_counts.items()
            }

        invalid_evidence = sum(
            1 for item in recommendations
            if not isinstance(item.get("evidence_ids"), list) or not item.get("evidence_ids")
        )

        return {
            "total": len(recommendations),
            "duplicates": duplicate_count,
            "invalid_evidence_rows": invalid_evidence,
            "zone_counts": dict(zone_counts),
            "missing_by_zone": missing_by_zone,
            "is_valid": duplicate_count == 0 and invalid_evidence == 0,
        }

    def count_matching(
        self,
        recommendations: list[dict[str, Any]],
        *,
        field: str | None,
        value: str,
    ) -> int:
        needle = self._normalise_text(value)
        return sum(1 for item in recommendations if needle and needle in self._search_text(item, field))

    def matches_filters(self, item: dict[str, Any], filters: dict[str, Any] | None) -> bool:
        if not filters:
            return True
        for field, value in filters.items():
            if value in (None, "", [], {}):
                continue
            values = value if isinstance(value, list) else [value]
            haystack = self._search_text(item, field)
            if not any(self._normalise_text(v) in haystack for v in values):
                return False
        return True

    def _search_text(self, item: dict[str, Any], field: str | None = None) -> str:
        aliases = {
            "location": ("location", "college", "reason"),
            "city": ("location", "college", "reason"),
            "district": ("location", "college", "reason"),
            "college": ("college",),
            "branch": ("branch",),
            "course": ("branch",),
            "zone": ("zone",),
            "seat_type": ("category_or_seat_type", "seat_allocation"),
            "category": ("category_or_seat_type",),
        }
        fields = aliases.get(str(field or "").casefold())
        if not fields:
            fields = (
                "college", "branch", "location", "zone",
                "category_or_seat_type", "seat_allocation", "reason",
            )
        return self._normalise_text(" ".join(str(item.get(name, "")) for name in fields))

    @staticmethod
    def _normalise_text(value: Any) -> str:
        text = re.sub(r"[^a-z0-9]+", " ", str(value or "").casefold())
        return " ".join(text.split())
