from __future__ import annotations

import re
from typing import Any, Callable

Record = dict[str, Any]
FilterReport = dict[str, Any]

# CAP category aliases are admission-domain rules. Actual values such as city,
# branch, college and university always come dynamically from the UI/profile.
_CATEGORY_TOKENS: dict[str, tuple[str, ...]] = {
    "OPEN": ("OPEN",),
    "OBC": ("OBC",),
    "SC": ("SC",),
    "ST": ("ST",),
    "EWS": ("EWS",),
    "SBC": ("SBC",),
    "VJ": ("VJ", "DT"),
    "VJDT": ("VJ", "DT"),
    "NTA": ("NTA",),
    "NTB": ("NTB",),
    "NTC": ("NTC",),
    "NTD": ("NTD",),
}

_INACTIVE_VALUES = {
    "",
    "any",
    "all",
    "none",
    "not specified",
    "no preference",
    "not available",
    "na",
    "n a",
}


def apply_profile_filters(
    results: list[Record],
    student_profile: dict[str, Any],
) -> tuple[list[Record], FilterReport]:
    """Apply active record-level UI filters before ranking.

    No college, city, branch or university is embedded in this tool. Selected
    values are read from ``student_profile`` and compared with indexed metadata.
    Empty/Any/No Preference values do not restrict results.

    Gender, category and HU/OHU are seat-code rules and are therefore evaluated
    later through :func:`seat_is_eligible` for each individual seat cutoff.
    """

    filtered = list(results)
    report: FilterReport = {
        "starting_rows": len(filtered),
        "active_filters": [],
        "steps": {},
    }

    pipeline: tuple[
        tuple[str, list[str], Callable[[Record], list[str]]], ...
    ] = (
        (
            "preferred_branches",
            _selected_values(
                student_profile.get("preferred_branches")
                or student_profile.get("preferred_branch")
            ),
            _branch_values,
        ),
        (
            "preferred_locations",
            _selected_values(student_profile.get("preferred_locations")),
            _location_values,
        ),
        (
            "seat_type",
            _selected_values(
                student_profile.get("seat_type")
                or student_profile.get("institute_type")
                or student_profile.get("college_type")
            ),
            _institution_type_values,
        ),
        (
            "college_preference",
            _selected_values(student_profile.get("college_preference")),
            _college_preference_values,
        ),
        (
            "preferred_colleges",
            _selected_values(
                student_profile.get("preferred_colleges")
                or student_profile.get("preferred_college")
            ),
            _college_values,
        ),
    )

    for name, expected, extractor in pipeline:
        if not expected:
            continue
        before = len(filtered)
        filtered = [
            record
            for record in filtered
            if _matches_any(extractor(record), expected)
        ]
        report["active_filters"].append(name)
        report["steps"][name] = {
            "requested": expected,
            "before": before,
            "after": len(filtered),
        }

    report["final_rows"] = len(filtered)
    return filtered, report


def seat_is_eligible(
    seat_type: str,
    student_profile: dict[str, Any],
    record: Record,
) -> bool:
    """Validate category, gender and university-area eligibility for one seat."""

    code = _seat_code(seat_type)
    if not code:
        return False

    if not _category_is_eligible(code, student_profile.get("category", "OPEN")):
        return False

    if not _gender_is_eligible(code, student_profile.get("gender")):
        return False

    if not _home_university_is_eligible(code, student_profile, record):
        return False

    return True


def _category_is_eligible(code: str, selected_category: Any) -> bool:
    # Special-purpose allocations are not ordinary category seats unless the
    # user explicitly selected that allocation through a future dedicated rule.
    if any(token in code for token in ("DEF", "PWD", "TFWS")):
        return False

    category = _compact(selected_category or "OPEN")
    tokens = _CATEGORY_TOKENS.get(category, (category,))
    return any(token and token in code for token in tokens)


def _gender_is_eligible(code: str, selected_gender: Any) -> bool:
    gender = _normalise(selected_gender)
    is_ladies_seat = code.startswith("L")

    if gender in {"female", "woman", "girl"}:
        # Female candidates may use both ladies and general seats.
        return True

    # Male/other/unspecified profiles must not receive ladies-only seats.
    return not is_ladies_seat


def _home_university_is_eligible(
    code: str,
    profile: dict[str, Any],
    record: Record,
) -> bool:
    selected = _single_value(profile.get("home_university"))
    if not selected:
        return True

    # State-level seats are valid irrespective of university area.
    if code.endswith("S"):
        return True

    record_universities = _university_values(record)
    if not record_universities:
        # Do not guess HU/OHU when indexed evidence lacks university metadata.
        return False

    same_university = _matches_any(record_universities, [selected])
    if same_university:
        return code.endswith("H")
    return code.endswith("O")


def _branch_values(record: Record) -> list[str]:
    metadata = _metadata(record)
    return _values(metadata, "branch", "course", "programme", "program")


def _location_values(record: Record) -> list[str]:
    metadata = _metadata(record)
    values = _values(
        metadata,
        "location",
        "city",
        "district",
        "place",
        "region",
        "college_location",
        "institute_location",
    )
    # Many CAP rows carry the city only inside the institute name. Matching the
    # selected location against that text is dynamic and does not hardcode cities.
    values.extend(_college_values(record))
    return values


def _institution_type_values(record: Record) -> list[str]:
    metadata = _metadata(record)
    return _values(
        metadata,
        "seat_type",
        "institute_type",
        "institution_type",
        "college_type",
        "ownership",
        "management",
        "status",
        "institute_status",
        "autonomous_status",
        "minority_status",
    )


def _college_preference_values(record: Record) -> list[str]:
    metadata = _metadata(record)
    values = _institution_type_values(record)
    values.extend(
        _values(
            metadata,
            "college_preference",
            "college_tier",
            "institute_tier",
            "tier",
            "ranking_group",
            "is_top_college",
        )
    )
    values.extend(_college_values(record))
    return values


def _college_values(record: Record) -> list[str]:
    metadata = _metadata(record)
    return _values(metadata, "college", "institute_name", "college_name", "institute")


def _university_values(record: Record) -> list[str]:
    metadata = _metadata(record)
    return _values(
        metadata,
        "home_university",
        "university",
        "university_name",
        "affiliating_university",
        "region_university",
    )


def _metadata(record: Record) -> dict[str, Any]:
    metadata = record.get("metadata")
    return metadata if isinstance(metadata, dict) else {}


def _values(metadata: dict[str, Any], *keys: str) -> list[str]:
    output: list[str] = []
    for key in keys:
        value = metadata.get(key)
        if value in (None, "", [], {}, ()):
            continue
        if isinstance(value, (list, tuple, set)):
            output.extend(str(item) for item in value if str(item).strip())
        else:
            output.append(str(value))
    return output


def _selected_values(value: Any) -> list[str]:
    if value in (None, "", [], {}, ()):
        return []
    items = value if isinstance(value, (list, tuple, set)) else [value]
    output: list[str] = []
    seen: set[str] = set()
    for item in items:
        text = " ".join(str(item or "").split())
        key = _normalise(text)
        if not key or key in _INACTIVE_VALUES or key in seen:
            continue
        output.append(text)
        seen.add(key)
    return output


def _single_value(value: Any) -> str | None:
    values = _selected_values(value)
    return values[0] if values else None


def _matches_any(actual_values: list[str], expected_values: list[str]) -> bool:
    actual = [_normalise(value) for value in actual_values if _normalise(value)]
    expected = [_normalise(value) for value in expected_values if _normalise(value)]
    if not expected:
        return True
    if not actual:
        return False

    for wanted in expected:
        wanted_tokens = set(wanted.split())
        for available in actual:
            available_tokens = set(available.split())
            if wanted == available or wanted in available or available in wanted:
                return True
            if wanted_tokens and wanted_tokens.issubset(available_tokens):
                return True
    return False


def _seat_code(value: Any) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(value or "").upper())


def _compact(value: Any) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(value or "").upper())


def _normalise(value: Any) -> str:
    return re.sub(
        r"\s+",
        " ",
        re.sub(r"[^a-z0-9]+", " ", str(value or "").casefold()),
    ).strip()