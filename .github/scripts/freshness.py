#!/usr/bin/env python3
"""Report topic folders that are due for review, based on README front matter.

A topic README starts with:

    ---
    status: current        # current | aging | archived
    last_reviewed: 2026-11-21
    review_every: 6mo      # e.g. 90d, 8w, 6mo, 1y
    ---

Every topic folder is expected to carry this metadata: each folder inside a
numbered category (01-foundations/mathematics/, ...) and any other top-level
folder. Other folders are tracked when their README.md has a last_reviewed key.
projects/ (mirrored from websites), archive/ and topics with status: archived
are skipped.

Prints a Markdown report to stdout. With --strict, exits 1 when any metadata
is invalid (used as a pull request check).
"""

import argparse
import calendar
import datetime as dt
import os
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import quote

DEFAULT_INTERVAL = "6mo"
DUE_SOON_DAYS = 30
STATUSES = {"current", "aging", "archived"}
SKIP_DIRS = {"archive", "node_modules", "projects"}
CATEGORY_RE = re.compile(r"^[0-9]{2}-")
INTERVAL_RE = re.compile(r"^(\d+)\s*(d|w|mo|y)$")


def front_matter(readme):
    lines = readme.read_text(encoding="utf-8", errors="replace").splitlines()
    if not lines or lines[0].strip() != "---":
        return {}
    meta = {}
    for line in lines[1:]:
        if line.strip() == "---":
            return meta
        key, sep, value = line.partition(":")
        if sep and key == key.strip() and not key.startswith("#"):
            meta[key] = value.split(" #")[0].strip().strip("'\"")
    meta["_error"] = "front matter is missing its closing `---`"
    return meta


def add_interval(start, interval):
    match = INTERVAL_RE.match(interval.strip().lower())
    if not match:
        raise ValueError(f"invalid review_every `{interval}` (use e.g. 90d, 8w, 6mo, 1y)")
    n, unit = int(match[1]), match[2]
    if unit == "d":
        return start + dt.timedelta(days=n)
    if unit == "w":
        return start + dt.timedelta(weeks=n)
    year, month = divmod(start.month - 1 + n * (12 if unit == "y" else 1), 12)
    year, month = start.year + year, month + 1
    return dt.date(year, month, min(start.day, calendar.monthrange(year, month)[1]))


def last_changed(root, path):
    try:
        result = subprocess.run(
            ["git", "log", "-1", "--format=%cs", "--follow", "--diff-filter=AM", "--", f"{path.as_posix()}/README.md"]
            if (root / path / "README.md").exists()
            else ["git", "log", "-1", "--format=%cs", "--", path.as_posix()],
            cwd=root, capture_output=True, text=True, check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def discover(root):
    """Map topic path (relative to root) -> front matter dict, or None if there is no README."""
    skip = lambda name: name in SKIP_DIRS or name.startswith(".")
    top = set()
    for p in root.iterdir():
        if p.is_dir() and not skip(p.name):
            children = [Path(p.name, c.name) for c in p.iterdir() if c.is_dir() and not skip(c.name)]
            top.update(children if CATEGORY_RE.match(p.name) else [Path(p.name)])
    topics = {p: None for p in top}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if not skip(d))
        if "README.md" in filenames:
            rel = Path(dirpath).relative_to(root)
            meta = front_matter(Path(dirpath) / "README.md")
            if rel in top or "last_reviewed" in meta:
                topics[rel] = meta
    return topics


def classify(root, topics, today):
    rows = {"overdue": [], "soon": [], "current": [], "archived": [], "invalid": [], "missing": []}
    for path, meta in sorted(topics.items()):
        if meta is None:
            rows["missing"].append((path, "no README.md", last_changed(root, path)))
            continue
        if "last_reviewed" not in meta:
            rows["missing"].append((path, "no review front matter", last_changed(root, path)))
            continue
        if "_error" in meta:
            rows["invalid"].append((path, meta["_error"]))
            continue
        status = meta.get("status", "current")
        if status == "archived":
            rows["archived"].append(path)
            continue
        interval = meta.get("review_every", DEFAULT_INTERVAL)
        try:
            if status not in STATUSES:
                raise ValueError(f"invalid status `{status}` (use {', '.join(sorted(STATUSES))})")
            try:
                reviewed = dt.date.fromisoformat(meta["last_reviewed"])
            except ValueError:
                raise ValueError(f"invalid last_reviewed `{meta['last_reviewed']}` (use YYYY-MM-DD)") from None
            if reviewed > today:
                raise ValueError(f"last_reviewed `{reviewed}` is in the future")
            due = add_interval(reviewed, interval)
        except ValueError as e:
            rows["invalid"].append((path, str(e)))
            continue
        bucket = "overdue" if due < today else "soon" if (due - today).days <= DUE_SOON_DAYS else "current"
        rows[bucket].append((path, reviewed, interval, due, status))
    return rows


def render(rows, base_url):
    def link(path):
        rel = path.as_posix()
        return f"[`{rel}`]({base_url}/{quote(rel)})" if base_url else f"`{rel}`"

    tracked = sum(len(rows[k]) for k in ("overdue", "soon", "current", "invalid"))
    out = [
        "## Freshness",
        "",
        f"{tracked} topics tracked: {len(rows['overdue'])} overdue, {len(rows['soon'])} due within "
        f"{DUE_SOON_DAYS} days, {len(rows['current'])} current, {len(rows['invalid'])} invalid. "
        f"{len(rows['archived'])} archived (skipped). {len(rows['missing'])} without review metadata.",
    ]
    for key, title in (("overdue", "Overdue"), ("soon", f"Due within {DUE_SOON_DAYS} days")):
        if rows[key]:
            out += ["", f"### {title}", "", "| Topic | Last reviewed | Every | Due | Status |", "|---|---|---|---|---|"]
            out += [f"| {link(p)} | {r} | {i} | {d} | {s} |" for p, r, i, d, s in sorted(rows[key], key=lambda x: x[3])]
    if rows["invalid"]:
        out += ["", "### Invalid metadata", "", "| Topic | Problem |", "|---|---|"]
        out += [f"| {link(p)} | {problem} |" for p, problem in rows["invalid"]]
    if rows["missing"]:
        out += ["", "### Missing review metadata", "",
                "Add front matter from `.github/TOPIC_TEMPLATE.md` after reviewing these (oldest first).",
                "", "| Topic | Problem | Last changed |", "|---|---|---|"]
        out += [f"| {link(p)} | {problem} | {changed} |" for p, problem, changed in sorted(rows["missing"], key=lambda x: x[2])]
    return "\n".join(out) + "\n"


def default_base_url():
    repo = os.environ.get("GITHUB_REPOSITORY")
    if not repo:
        return ""
    server = os.environ.get("GITHUB_SERVER_URL", "https://github.com")
    return f"{server}/{repo}/tree/{os.environ.get('GITHUB_REF_NAME') or 'main'}"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--today", type=dt.date.fromisoformat, default=dt.datetime.now(dt.timezone.utc).date())
    parser.add_argument("--base-url", default=default_base_url(), help="prefix for topic links in the report")
    parser.add_argument("--strict", action="store_true", help="exit 1 if any metadata is invalid")
    args = parser.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")
    root = args.root.resolve()
    rows = classify(root, discover(root), args.today)
    print(render(rows, args.base_url.rstrip("/")), end="")
    if args.strict and rows["invalid"]:
        for path, problem in rows["invalid"]:
            print(f"::error file={path.as_posix()}/README.md::{problem}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
