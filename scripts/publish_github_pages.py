#!/usr/bin/env python3
"""Publish the documentation for one vX.Y.Z tag onto GitHub Pages.

``prepare`` reads the existing gh-pages directories and writes that version
menu into this checkout's docs/conf.py. The CI job discards that checkout
after the build, so the change is not pushed back to GitLab.

``publish`` copies one Sphinx HTML tree to gh-pages as both ``vX.Y.Z`` and
``latest``, refreshes the shared changelog, and adds the new version to the
menus of pages already on the site.
"""

import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = "https://github.com/ChASE-library/ChASE.git"
TAG_RE = re.compile(r"^v\d+\.\d+\.\d+$")
VERSION_DIR_RE = re.compile(r"^v\d+\.\d+\.\d+$")
VERSIONS_DICT_RE = re.compile(r"'versions':\s*\{[^}]*\}", re.DOTALL)
LATEST_OPTION_RE = re.compile(
    r'(<option value="/ChASE/latest/">\s*latest\s*</option>)'
)
MENU_RE = re.compile(
    r'<div class="wy-menu wy-menu-vertical".*?</div>',
    re.DOTALL,
)


def release_tag():
    tag = os.environ.get("CI_COMMIT_TAG") or os.environ.get("DOCS_VERSION", "")
    if not TAG_RE.fullmatch(tag):
        sys.exit("Set CI_COMMIT_TAG or DOCS_VERSION to a tag such as v1.8.0.")
    return tag


def version_key(tag):
    return tuple(int(part) for part in tag[1:].split("."))


def clone_pages(dest):
    subprocess.check_call(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "--branch",
            "gh-pages",
            REPO,
            str(dest),
        ]
    )


def installed_versions(pages_root):
    found = [
        path.name
        for path in pages_root.iterdir()
        if path.is_dir() and VERSION_DIR_RE.fullmatch(path.name)
    ]
    return sorted(found, key=version_key, reverse=True)


def write_conf_versions(tags):
    lines = [
        "'versions': {",
        "        'changelog': '/ChASE/changelog/',",
        "        'latest': '/ChASE/latest/',",
    ]
    for tag in tags:
        lines.append(f"        '{tag}': '/ChASE/{tag}/',")
    lines.append("    }")
    conf = ROOT / "docs" / "conf.py"
    text = conf.read_text()
    updated, count = VERSIONS_DICT_RE.subn("\n".join(lines), text, count=1)
    if count != 1:
        sys.exit("Could not update the version menu in docs/conf.py.")
    conf.write_text(updated)


def prepare():
    tag = release_tag()
    with tempfile.TemporaryDirectory(prefix="chase-pages-") as tmp:
        dest = Path(tmp) / "gh-pages"
        clone_pages(dest)
        tags = installed_versions(dest)
    if tag not in tags:
        tags.append(tag)
        tags = sorted(tags, key=version_key, reverse=True)
    write_conf_versions(tags)
    print("Version menu:", ", ".join(tags))


def retarget_changelog_sidebar(html):
    def rewrite_menu(match):
        menu = match.group(0)

        def rewrite_href(href_match):
            href = href_match.group(1)
            if href.startswith(("#", "/", "http", "mailto:")):
                return href_match.group(0)
            return f'href="/ChASE/latest/{href}"'

        menu = re.sub(r'href="([^"]*)"', rewrite_href, menu)
        menu = re.sub(
            r'<li class="toctree-l1 current"><a class="current reference internal" href="(?:#|/ChASE/latest/changelog\.html)">Changelog</a>',
            '<li class="toctree-l1 current"><a class="reference internal current" href="/ChASE/changelog/">Changelog</a>',
            menu,
            count=1,
        )
        menu = re.sub(
            r'<li class="toctree-l1"><a class="reference internal" href="/ChASE/latest/changelog\.html">Changelog</a>',
            '<li class="toctree-l1 current"><a class="reference internal current" href="/ChASE/changelog/">Changelog</a>',
            menu,
            count=1,
        )
        return menu

    updated, count = MENU_RE.subn(rewrite_menu, html, count=1)
    menu = MENU_RE.search(updated)
    if (
        count != 1
        or menu is None
        or 'href="/ChASE/changelog/"' not in menu.group(0)
    ):
        sys.exit("Could not build the shared changelog page.")
    return updated


def install_changelog(pages_root, tag):
    source = pages_root / tag / "changelog.html"
    if not source.is_file():
        sys.exit(f"The {tag} documentation has no changelog.html.")
    page = retarget_changelog_sidebar(source.read_text(errors="replace"))
    dest = pages_root / "changelog"
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir()
    for extra in ("_static", "_images"):
        extra_source = pages_root / tag / extra
        if extra_source.is_dir():
            shutil.copytree(extra_source, dest / extra)
    (dest / "index.html").write_text(page)
    (dest / "changelog.html").write_text(page)


def add_version_option(html, tag):
    option = f'<option value="/ChASE/{tag}/">\n      {tag}\n    </option>'
    if f'value="/ChASE/{tag}/"' in html or 'id="version-select"' not in html:
        return html, False
    updated, count = LATEST_OPTION_RE.subn(r"\1" + option, html, count=1)
    if count != 1:
        return html, True
    return updated, False


def patch_existing_menus(pages_root, tag):
    failures = []
    for path in pages_root.rglob("*.html"):
        if tag in path.parts or "latest" in path.parts or "changelog" in path.parts:
            continue
        text = path.read_text(errors="replace")
        updated, failed = add_version_option(text, tag)
        if failed:
            failures.append(str(path.relative_to(pages_root)))
            continue
        if updated != text:
            path.write_text(updated)
    if failures:
        print("Could not add the new version to these pages:")
        for name in failures:
            print(f"  {name}")
        sys.exit(1)


def write_index(pages_root, tag):
    tags = installed_versions(pages_root)
    lines = [
        "<!DOCTYPE html>",
        "<html>",
        "<head>",
        '    <meta charset="utf-8">',
        '    <meta http-equiv="refresh" content="0; url=latest/">',
        '    <link rel="canonical" href="latest/" />',
        "    <title>ChASE Documentation</title>",
        "</head>",
        "<body>",
        f'    <p>Redirecting to <a href="latest/">latest documentation</a> ({tag}).</p>',
        "    <ul>",
        '    <li><a href="changelog/">Changelog</a></li>',
        f'    <li><a href="latest/">latest</a> ({tag})</li>',
    ]
    for name in tags:
        lines.append(f'    <li><a href="{name}/">{name}</a></li>')
    lines.extend(["    </ul>", "</body>", "</html>", ""])
    (pages_root / "index.html").write_text("\n".join(lines))


def commit_and_push(pages_root, tag):
    token = os.environ.get("GH_PAGES_TOKEN", "")
    if not token:
        sys.exit("Set the GitLab CI variable GH_PAGES_TOKEN.")
    subprocess.check_call(["git", "config", "user.name", "ChASE Bot"], cwd=pages_root)
    subprocess.check_call(
        ["git", "config", "user.email", "chase@fz-juelich.de"],
        cwd=pages_root,
    )
    subprocess.check_call(["git", "add", "-A"], cwd=pages_root)
    diff = subprocess.run(
        ["git", "diff", "--cached", "--quiet"],
        cwd=pages_root,
    )
    if diff.returncode == 0:
        print("Documentation is already published.")
        return
    subprocess.check_call(
        ["git", "commit", "-m", f"Publish documentation for {tag}."],
        cwd=pages_root,
    )
    askpass_dir = tempfile.mkdtemp(prefix="chase-askpass-")
    askpass = Path(askpass_dir) / "askpass"
    askpass.write_text('#!/bin/sh\nprintf "%s\\n" "$GH_PAGES_TOKEN"\n')
    askpass.chmod(0o700)
    env = os.environ.copy()
    env["GIT_ASKPASS"] = str(askpass)
    env["GIT_TERMINAL_PROMPT"] = "0"
    try:
        subprocess.check_call(
            [
                "git",
                "push",
                "https://x-access-token@github.com/ChASE-library/ChASE.git",
                "HEAD:gh-pages",
            ],
            cwd=pages_root,
            env=env,
        )
    finally:
        shutil.rmtree(askpass_dir, ignore_errors=True)


def apply_release(dest, html, tag):
    if not (html / "index.html").is_file():
        sys.exit(f"No Sphinx output at {html}.")
    for name in (tag, "latest"):
        target = dest / name
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(html, target)
    install_changelog(dest, tag)
    patch_existing_menus(dest, tag)
    write_index(dest, tag)
    (dest / ".nojekyll").touch()
    stray = dest / "changelog.html"
    if stray.is_file():
        stray.unlink()


def stage(html_dir, pages_dir):
    apply_release(Path(pages_dir), Path(html_dir), release_tag())
    print(f"Staged the site in {pages_dir}. Nothing was pushed.")


def publish(html_dir):
    tag = release_tag()
    with tempfile.TemporaryDirectory(prefix="chase-pages-") as tmp:
        dest = Path(tmp) / "gh-pages"
        clone_pages(dest)
        apply_release(dest, Path(html_dir), tag)
        commit_and_push(dest, tag)
    print(f"Published https://chase-library.github.io/ChASE/{tag}/")


def main():
    if len(sys.argv) == 2 and sys.argv[1] == "prepare":
        prepare()
        return
    if len(sys.argv) == 3 and sys.argv[1] == "publish":
        publish(sys.argv[2])
        return
    if len(sys.argv) == 4 and sys.argv[1] == "stage":
        stage(sys.argv[2], sys.argv[3])
        return
    sys.exit("Usage: publish_github_pages.py prepare|publish HTML_DIR|stage HTML_DIR PAGES_DIR")


if __name__ == "__main__":
    main()
