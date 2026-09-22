"""Fail the release if a built distribution carries an AWS account id.

The 0.1.3 incident: an sdist ships files a wheel does not -- ``bench/``, ``examples/``,
demo scripts -- so an account id written into a code comment reached PyPI even though it
was never part of the importable package. Linting the source tree would not have caught
it either, because the id was removed from the tree only after 0.1.3 was built. The guard
therefore runs against the artifact, which is the thing that actually gets published.

Detection is a 12-digit run at a non-digit boundary, the shape of an AWS account id. The
obviously-fake ``123456789012`` used in documentation is allowed so examples can keep
showing a realistic ARN.

Usage::

    python scripts/scan_artifacts.py dist/*
"""

from __future__ import annotations

import re
import sys
import tarfile
import zipfile
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

ACCOUNT_ID: Final = re.compile(r"(?<![0-9])[0-9]{12}(?![0-9])")

# Placeholders that are meant to appear in docs and samples.
ALLOWED: Final[frozenset[str]] = frozenset({"123456789012", "000000000000"})

# Binary members are not worth decoding; an account id lands in text.
SKIP_SUFFIXES: Final[frozenset[str]] = frozenset(
    {".so", ".pyd", ".dylib", ".whl", ".png", ".jpg", ".gif", ".ico"}
)


@dataclass(frozen=True, slots=True)
class Finding:
    """One offending value in one member of one distribution."""

    archive: str
    member: str
    value: str


def _is_text(name: str) -> bool:
    return not any(name.endswith(s) for s in SKIP_SUFFIXES)


def _members(path: Path) -> Iterator[tuple[str, bytes]]:
    """Yield (member name, raw bytes) for every text member of a wheel or sdist."""
    if path.suffix == ".whl" or path.suffix == ".zip":
        with zipfile.ZipFile(path) as z:
            for name in z.namelist():
                if _is_text(name):
                    yield name, z.read(name)
        return
    with tarfile.open(path) as t:
        for member in t.getmembers():
            if not member.isfile() or not _is_text(member.name):
                continue
            fh = t.extractfile(member)
            if fh is None:
                continue
            yield member.name, fh.read()


def scan(path: Path) -> tuple[Finding, ...]:
    """Return every account id found in ``path``. Empty means clean."""
    found: list[Finding] = []
    for name, raw in _members(path):
        text = raw.decode("utf-8", "ignore")
        for match in ACCOUNT_ID.finditer(text):
            value = match.group()
            if value not in ALLOWED:
                found.append(Finding(archive=path.name, member=name, value=value))
    return tuple(found)


def main(argv: Sequence[str]) -> int:
    """Scan each path given on the command line. Non-zero exit means a leak."""
    paths = [Path(a) for a in argv]
    if not paths:
        print("usage: scan_artifacts.py dist/*", file=sys.stderr)
        return 2

    findings: list[Finding] = []
    for path in paths:
        if not path.is_file():
            print(f"  skip (not a file): {path}")
            continue
        hits = scan(path)
        status = f"{len(hits)} FINDING(S)" if hits else "clean"
        print(f"  {path.name}: {status}")
        findings.extend(hits)

    if not findings:
        print("No AWS account ids in any distribution.")
        return 0

    print("\nAWS account id found in a distribution about to be published:", file=sys.stderr)
    for f in findings:
        # Print the member, never the value -- this output lands in public CI logs.
        print(f"  {f.archive}: {f.member}", file=sys.stderr)
    print(
        "\nRemove it, rebuild, and release again. Record the region and date only, or "
        "read the account at runtime. See the No AWS Account IDs rule.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
