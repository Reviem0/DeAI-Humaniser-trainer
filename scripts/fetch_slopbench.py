"""Put the SlopBench corpus in data/slopbench, where deai.data reads it.

SlopBench ships with SlopTotal (MIT, https://github.com/pablocaeg/sloptotal) in
tests/eval/slopbench/corpus. The files are pinned to one upstream commit:
deai.data hashes passages into splits, so a changed corpus would change every
split and invalidate every run and baseline.

    .venv\\Scripts\\python scripts\\fetch_slopbench.py                 # download
    .venv\\Scripts\\python scripts\\fetch_slopbench.py --from D:\\AI\\sloptotal\\tests\\eval\\slopbench\\corpus
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from deai.data import SLOPBENCH  # noqa: E402

REPO = "pablocaeg/sloptotal"
COMMIT = "3d5a3577a83cd43604699ee4c445a0145a8df359"
CORPUS = "tests/eval/slopbench/corpus"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from", dest="src", type=Path, default=None, help="copy from a local corpus folder instead")
    ap.add_argument("--out", type=Path, default=SLOPBENCH)
    ap.add_argument("--commit", default=COMMIT)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    if args.src:
        files = sorted(args.src.glob("*.jsonl"))
        for f in files:
            shutil.copyfile(f, args.out / f.name)
        print(f"copied {len(files)} files from {args.src} to {args.out}")
        return 0

    with httpx.Client(timeout=60, follow_redirects=True) as http:
        r = http.get(f"https://api.github.com/repos/{REPO}/contents/{CORPUS}", params={"ref": args.commit})
        r.raise_for_status()
        names = sorted(e["name"] for e in r.json() if e["name"].endswith(".jsonl"))
        for name in names:
            body = http.get(f"https://raw.githubusercontent.com/{REPO}/{args.commit}/{CORPUS}/{name}")
            body.raise_for_status()
            (args.out / name).write_bytes(body.content)
            print(f"  {name} ({len(body.content) / 1024:.0f} KB)")
    print(f"downloaded {len(names)} files from {REPO}@{args.commit[:7]} to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
