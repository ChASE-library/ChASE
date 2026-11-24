#!/usr/bin/env python3
"""Keep the committed release strings aligned with CMakeLists.txt.

The CMake project version is the number already chosen for this tree.
This script does not decide whether the next release is a major, minor,
or patch. A person passes that number to ``set``.

``check`` fails when these disagree with project(VERSION):

- CITATION.cff ``version``
- README.md ``git checkout``
- docs/conf.py ``version`` and ``release``
- docs/quick-start.rst ``git checkout``

On a version tag, ``check`` also requires the tag's X.Y.Z to match CMake.
"""

import argparse
import datetime
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION_RE = re.compile(r"(\d+\.\d+\.\d+)")
TAG_RE = re.compile(r"^v(\d+\.\d+\.\d+)(?:-rc\d+)?$")

CMAKE = ROOT / "CMakeLists.txt"
CITATION = ROOT / "CITATION.cff"
README = ROOT / "README.md"
CONF = ROOT / "docs" / "conf.py"
QUICK_START = ROOT / "docs" / "quick-start.rst"

CMAKE_LINE = re.compile(
    r"^project\(\s*ChASE\s+LANGUAGES\s+C\s+CXX\s+Fortran\s+VERSION\s+(\d+\.\d+\.\d+)\s*\)\s*$",
    re.MULTILINE,
)
CITATION_VERSION = re.compile(r"^version:\s+v(\d+\.\d+\.\d+)\s*$", re.MULTILINE)
CITATION_DATE = re.compile(r"^date-released:\s+'(\d{4}-\d{2}-\d{2})'\s*$", re.MULTILINE)
README_CHECKOUT = re.compile(r"^(git checkout )v(\d+\.\d+\.\d+)( #.*)$", re.MULTILINE)
CONF_VERSION = re.compile(r"^version = u'v(\d+\.\d+\.\d+)'\s*$", re.MULTILINE)
CONF_RELEASE = re.compile(r"^release = u'v(\d+\.\d+\.\d+)'\s*$", re.MULTILINE)
QUICK_CHECKOUT = re.compile(r"^(    git checkout )v(\d+\.\d+\.\d+)\s*$", re.MULTILINE)


def read(path):
    return path.read_text(encoding="utf-8")


def write(path, text):
    if not text.endswith("\n"):
        text += "\n"
    path.write_text(text, encoding="utf-8")


def one_match(pattern, text, label):
    found = pattern.findall(text)
    if len(found) != 1:
        raise SystemExit(f"{label}: expected 1 version, found {len(found)}")
    value = found[0]
    if isinstance(value, tuple):
        return value[1]
    return value


def cmake_version(text):
    return one_match(CMAKE_LINE, text, "CMakeLists.txt")


def collected_versions():
    return {
        "CMakeLists.txt": cmake_version(read(CMAKE)),
        "CITATION.cff": one_match(CITATION_VERSION, read(CITATION), "CITATION.cff"),
        "README.md": one_match(README_CHECKOUT, read(README), "README.md"),
        "docs/conf.py version": one_match(CONF_VERSION, read(CONF), "docs/conf.py version"),
        "docs/conf.py release": one_match(CONF_RELEASE, read(CONF), "docs/conf.py release"),
        "docs/quick-start.rst": one_match(QUICK_CHECKOUT, read(QUICK_START), "docs/quick-start.rst"),
    }


def parse_version(raw):
    text = raw.strip()
    if text.startswith("v"):
        text = text[1:]
    if not re.fullmatch(r"\d+\.\d+\.\d+", text):
        raise SystemExit(f"Version must look like 1.8.0 or v1.8.0, got {raw!r}")
    return text


def release_date(explicit):
    if explicit:
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", explicit):
            raise SystemExit(f"Date must be YYYY-MM-DD, got {explicit!r}")
        return explicit
    stamped = os.environ.get("CI_PIPELINE_CREATED_AT", "")
    if len(stamped) >= 10 and re.fullmatch(r"\d{4}-\d{2}-\d{2}", stamped[:10]):
        return stamped[:10]
    return datetime.date.today().isoformat()


def replace_one(pattern, text, repl, label):
    new_text, count = pattern.subn(repl, text)
    if count != 1:
        raise SystemExit(f"{label}: expected to replace 1 line, replaced {count}")
    return new_text


def command_check():
    versions = collected_versions()
    base = versions["CMakeLists.txt"]
    mismatched = [f"  {name}: {value}" for name, value in versions.items() if value != base]
    if mismatched:
        print("Release version strings disagree with CMakeLists.txt:")
        print("\n".join(mismatched))
        print("Run scripts/sync_release_version.py set <version> with the version you chose.")
        return 1

    tag = os.environ.get("CI_COMMIT_TAG", "")
    matched = TAG_RE.fullmatch(tag) if tag else None
    if matched and matched.group(1) != base:
        print(f"Tag {tag} does not match CMake version {base}.")
        return 1

    print(f"Release version strings match {base}.")
    if tag:
        print(f"Tag {tag} matches that version.")
    return 0


def command_set(raw_version, raw_date):
    version = parse_version(raw_version)
    date = release_date(raw_date)

    write(CMAKE, replace_one(
        CMAKE_LINE,
        read(CMAKE),
        f"project( ChASE LANGUAGES C CXX Fortran VERSION {version} )",
        "CMakeLists.txt",
    ))
    citation = replace_one(
        CITATION_VERSION,
        read(CITATION),
        f"version: v{version}",
        "CITATION.cff version",
    )
    citation = replace_one(
        CITATION_DATE,
        citation,
        f"date-released: '{date}'",
        "CITATION.cff date-released",
    )
    write(CITATION, citation)
    write(README, replace_one(
        README_CHECKOUT,
        read(README),
        rf"\1v{version}\3",
        "README.md",
    ))
    conf = replace_one(
        CONF_VERSION,
        read(CONF),
        f"version = u'v{version}'",
        "docs/conf.py version",
    )
    conf = replace_one(
        CONF_RELEASE,
        conf,
        f"release = u'v{version}'",
        "docs/conf.py release",
    )
    write(CONF, conf)
    write(QUICK_START, replace_one(
        QUICK_CHECKOUT,
        read(QUICK_START),
        rf"\1v{version}",
        "docs/quick-start.rst",
    ))
    print(f"Set release version to {version}, date-released {date}.")
    return command_check()


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("check", help="compare the committed version strings")
    set_parser = commands.add_parser("set", help="write a version chosen by the caller")
    set_parser.add_argument("version", help="for example 1.8.0")
    set_parser.add_argument("--date", help="CITATION.cff date-released, YYYY-MM-DD")
    args = parser.parse_args(argv)
    if args.command == "check":
        return command_check()
    return command_set(args.version, args.date)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
