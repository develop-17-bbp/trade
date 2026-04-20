"""Stdlib-only ULID generator — no external dep.

Drops the `python-ulid` dependency that was colliding with another
package named `ulid` on the GPU box (`from ulid import ULID` raised
ImportError because the sibling package ships as a single module,
not a package with a ULID class).

Format (Crockford Base32):
    26 chars, 10 for 48-bit millisecond timestamp, 16 for 80 bits
    of randomness. Time-sortable lexicographically.
    Spec: https://github.com/ulid/spec
"""

from __future__ import annotations

import os
import time

# Crockford Base32 — excludes I, L, O, U to avoid ambiguity
_CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


def _encode(value: int, length: int) -> str:
    """Encode ``value`` as Crockford Base32, zero-padded to ``length``."""
    if value < 0:
        raise ValueError("value must be non-negative")
    if length < 1:
        raise ValueError("length must be positive")
    out = []
    for _ in range(length):
        out.append(_CROCKFORD[value & 0x1F])
        value >>= 5
    if value != 0:
        # Encoded value exceeded the requested length.
        raise ValueError("value does not fit in requested length")
    return "".join(reversed(out))


def new_ulid() -> str:
    """Return a fresh 26-char Crockford Base32 ULID string.

    Layout:
      - first 10 chars: 48-bit millisecond timestamp
      - last 16 chars: 80 bits of os.urandom-backed randomness

    The timestamp portion guarantees time-sortability at ms granularity.
    Within the same millisecond, the 80 random bits provide collision
    resistance (2^80 ≈ 10^24 values).
    """
    ts_ms = int(time.time() * 1000) & ((1 << 48) - 1)
    randomness = int.from_bytes(os.urandom(10), "big") & ((1 << 80) - 1)
    return _encode(ts_ms, 10) + _encode(randomness, 16)


def is_ulid(s: str) -> bool:
    """True if ``s`` looks like a Crockford-Base32 ULID (26 chars, valid alphabet)."""
    if not isinstance(s, str) or len(s) != 26:
        return False
    return all(c in _CROCKFORD for c in s)
