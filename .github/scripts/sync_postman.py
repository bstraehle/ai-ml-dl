#!/usr/bin/env python3
"""Export every collection in the public Generative AI APIs Postman workspace into projects/rest-apis/.

Each collection is written as Postman v2.1 JSON (rest-apis/<name>.postman_collection.json),
other files in rest-apis/ are removed, and rest-apis/README.md is regenerated. Only reads from
Postman. Needs a Postman API key in POSTMAN_API_KEY (or in the file given by --env-file).
Nothing is written if a collection looks like it contains an API key.

    python .github/scripts/sync_postman.py [--env-file path/to/.env]
"""

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

API = "https://api.getpostman.com"
WORKSPACE_ID = "39776787-fc1a-4934-8039-ec1c0147df80"
WORKSPACE_URL = "https://www.postman.com/ai-engineer/generative-ai-apis/"
SECRET_RE = re.compile(
    r"sk-(?:proj-|ant-)?[A-Za-z0-9_-]{32,}|hf_[A-Za-z0-9]{30,}|AKIA[0-9A-Z]{16}|AIza[0-9A-Za-z_-]{35}"
    r"|xai-[A-Za-z0-9]{40,}|gsk_[A-Za-z0-9]{40,}|r8_[A-Za-z0-9]{30,}|PMAK-[0-9a-f]{24}-[0-9a-f]{34}"
)


def api(path, key):
    req = urllib.request.Request(API + path, headers={"X-Api-Key": key, "Accept": "application/json"})
    for delay in (5, 15, 30, 60, 0):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code != 429 or not delay:
                sys.exit(f"Postman API GET {path} -> {e.code}")
            time.sleep(delay)


def load_key(env_file):
    if env_file and not os.environ.get("POSTMAN_API_KEY"):
        for line in Path(env_file).read_text(encoding="utf-8").splitlines():
            name, sep, value = line.partition("=")
            if sep and name.strip() == "POSTMAN_API_KEY":
                os.environ["POSTMAN_API_KEY"] = value.strip().strip("'\"")
    return os.environ.get("POSTMAN_API_KEY") or sys.exit("POSTMAN_API_KEY is not set")


def display_name(name):
    return name.replace("📌", "").strip()


def file_name(name):
    return re.sub(r"[^a-z0-9.]+", "-", display_name(name).lower()).strip("-") + ".postman_collection.json"


def count_requests(items):
    return sum(count_requests(i["item"]) if "item" in i else 1 for i in items)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dest", type=Path, default=Path("projects/rest-apis"))
    parser.add_argument("--env-file", help="read POSTMAN_API_KEY from this .env file")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    key = load_key(args.env_file)

    listed = api(f"/collections?workspace={WORKSPACE_ID}", key)["collections"]
    exports = {}
    for c in sorted(listed, key=lambda c: display_name(c["name"]).casefold()):
        collection = api(f"/collections/{c['uid']}", key)["collection"]
        text = json.dumps(collection, indent=2, ensure_ascii=False) + "\n"
        if SECRET_RE.search(text):
            sys.exit(f"possible API key in '{c['name']}'; nothing was written")
        exports[file_name(c["name"])] = (c["name"], count_requests(collection.get("item", [])), text)
        print(f"{display_name(c['name'])}: {exports[file_name(c['name'])][1]} requests")

    args.dest.mkdir(exist_ok=True)
    for p in args.dest.iterdir():
        if p.is_file() and p.name != "README.md" and p.name not in exports:
            p.unlink()
    for fname, (_, _, text) in exports.items():
        (args.dest / fname).write_text(text, encoding="utf-8")

    rows = "\n".join(f"| {display_name(n)} | {count} | [{f}]({f}) |" for f, (n, count, _) in exports.items())
    (args.dest / "README.md").write_text(
        "# Generative AI REST APIs\n\n"
        f"Postman collections for {len(exports)} generative AI providers, exported from the public "
        f"[Generative AI APIs]({WORKSPACE_URL}) workspace. Most start with a Get Started request that has a "
        "saved example response from the real API.\n\n"
        "To use one, fork it from the workspace, or import its JSON file into Postman. Collections read API keys "
        "from your Postman Vault (`{{vault:<provider>_api_key}}`), so no keys are stored in them. Local-only "
        "APIs need no key.\n\n"
        "Don't edit these files here: change the workspace, then run `python .github/scripts/sync_postman.py`.\n\n"
        f"| Collection | Requests | File |\n|---|---|---|\n{rows}\n",
        encoding="utf-8",
    )
    print(f"exported {len(exports)} collections to {args.dest}/")


if __name__ == "__main__":
    main()
