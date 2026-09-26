"""Shared, domain-independent helpers."""

from __future__ import annotations


def identifier_key(identifier: str) -> tuple[str, int, str]:
    """Sort numeric ID suffixes naturally within each prefix."""
    prefix = identifier.rstrip("0123456789")
    suffix = identifier[len(prefix) :]
    return (prefix, int(suffix) if suffix else -1, identifier)
