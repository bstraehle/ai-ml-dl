#!/usr/bin/env python3
"""Mirror an author's public Hugging Face Spaces into projects/hugging-face/.

Each Space becomes projects/hugging-face/<space>/ with the files of its latest commit.
Files that no longer exist in the Space are deleted, and so are folders for
Spaces that are gone or private. projects/hugging-face/README.md is regenerated as an
index. Needs no credentials.

    python .github/scripts/sync_hf_spaces.py [--author bstraehle] [--dry-run]
"""

import argparse
import json
import shutil
import sys
import urllib.request
from pathlib import Path
from urllib.parse import quote

HUB = "https://huggingface.co"
ALWAYS_SKIP = {".gitattributes"}
SKIP_PREFIXES = {"gaia": ("files/",)}
MAX_BYTES = 5_000_000
NOTES = {
    "gaia": "Dependabot reports litellm advisories for `gaia/requirements.txt`. litellm stays at 1.80.0 because "
            "stagehand 0.5.x (the last release that the crewai-tools Stagehand browser tool supports) caps it there. "
            "Most advisories concern the LiteLLM Proxy server, which this Space doesn't run; it uses litellm only "
            "as a library.",
}


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "ai-ml-dl-sync"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def unify(data):
    return data.replace(b"\r\n", b"\n")


def public_spaces(author):
    spaces = json.loads(get(f"{HUB}/api/spaces?author={author}&full=true&limit=1000"))
    return sorted((s for s in spaces if not s.get("private")), key=lambda s: s["id"])


def space_files(space_id, sha):
    tree = json.loads(get(f"{HUB}/api/spaces/{space_id}/tree/{sha}?recursive=true"))
    return [f for f in tree if f["type"] == "file"]


def skip_reason(name, f):
    if f["path"] in ALWAYS_SKIP:
        return "git config"
    if f["path"].startswith(SKIP_PREFIXES.get(name, ())):
        return "excluded"
    if f["size"] > MAX_BYTES:
        return f"{f['size'] // 1_000_000} MB"
    return None


def sync_space(space, dest, dry_run):
    name = space["id"].split("/", 1)[1]
    target = dest / name
    kept, skipped, changed = set(), [], 0
    for f in space_files(space["id"], space["sha"]):
        reason = skip_reason(name, f)
        if reason:
            skipped.append((f["path"], reason))
            continue
        kept.add(f["path"])
        path = target / f["path"]
        data = get(f"{HUB}/spaces/{space['id']}/resolve/{space['sha']}/{quote(f['path'])}")
        if not path.exists() or unify(path.read_bytes()) != unify(data):
            changed += 1
            if not dry_run:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
    stale = [p for p in target.rglob("*") if p.is_file() and p.relative_to(target).as_posix() not in kept] if target.exists() else []
    for p in stale:
        if not dry_run:
            p.unlink()
    if not dry_run and target.exists():
        for d in sorted((d for d in target.rglob("*") if d.is_dir()), key=lambda d: len(d.parts), reverse=True):
            if not any(d.iterdir()):
                d.rmdir()
    print(f"{name}: {len(kept)} files, {changed} added or changed, {len(stale)} removed"
          + (f", skipped {len(skipped)} ({', '.join(sorted({r for _, r in skipped}))})" if skipped else ""))
    return name, skipped


def write_index(dest, author, spaces, skipped):
    rows = []
    for s in spaces:
        name = s["id"].split("/", 1)[1]
        card = s.get("cardData") or {}
        note = " (benchmark files aren't mirrored; see the Space)" if any(r == "excluded" for _, r in skipped.get(name, [])) else ""
        rows.append(f"| [{card.get('title', name)}]({HUB}/spaces/{s['id']}) | {card.get('short_description', '')}{note} | [{name}/]({name}/) |")
    (dest / "README.md").write_text(
        "# Hugging Face Spaces\n\n"
        f"Source code for my public [Hugging Face Spaces]({HUB}/{author}), mirrored from the Hub.\n"
        "Don't edit these folders here: change the Space, then run `python .github/scripts/sync_hf_spaces.py`.\n\n"
        "| Space | What it does | Source |\n|---|---|---|\n" + "\n".join(rows) + "\n"
        + "".join(f"\n**{name}:** {text}\n" for name, text in NOTES.items() if any(s["id"].endswith("/" + name) for s in spaces)),
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--author", default="bstraehle")
    parser.add_argument("--dest", type=Path, default=Path("projects/hugging-face"))
    parser.add_argument("--dry-run", action="store_true", help="report changes without writing")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    spaces = public_spaces(args.author)
    skipped = dict(sync_space(s, args.dest, args.dry_run) for s in spaces)
    names = {s["id"].split("/", 1)[1] for s in spaces}
    for p in sorted(args.dest.iterdir()):
        if p.name != "README.md" and p.name not in names:
            print(f"removing {p.name}: no public Space")
            if not args.dry_run:
                shutil.rmtree(p) if p.is_dir() else p.unlink()
    if not args.dry_run:
        write_index(args.dest, args.author, spaces, skipped)


if __name__ == "__main__":
    main()
