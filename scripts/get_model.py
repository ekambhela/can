#!/usr/bin/env python3
"""Ensure a trained model exists at artifacts/model.joblib.

The binary is no longer committed to git (see AUDIT.md item 3). This script is
the one-command way to materialize it:

    python scripts/get_model.py            # train from the committed GDSC data
    python scripts/get_model.py --force    # retrain even if it already exists

To fetch a pre-built artifact instead of training (e.g. a GitHub Release asset or
an S3/HF-Hub object), set MODEL_URL and it will be downloaded rather than trained.
This keeps deploys fast without versioning the binary in git.

    MODEL_URL=https://example.com/model.joblib python scripts/get_model.py
"""

from __future__ import annotations

import argparse
import os
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS = os.path.join(ROOT, "artifacts")
MODEL_PATH = os.path.join(ARTIFACTS, "model.joblib")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true", help="retrain/redownload even if present")
    args = ap.parse_args()

    if os.path.exists(MODEL_PATH) and not args.force:
        print(f"model already present: {MODEL_PATH}")
        return 0

    os.makedirs(ARTIFACTS, exist_ok=True)
    url = os.environ.get("MODEL_URL")
    if url:
        print(f"downloading model from {url} ...")
        urllib.request.urlretrieve(url, MODEL_PATH)  # noqa: S310 — operator-supplied URL
        print(f"saved {MODEL_PATH}")
        return 0

    # No URL: train from the committed data. Import lazily so --help needs no deps.
    sys.path.insert(0, ROOT)
    from model.train import main as train_main

    print("no MODEL_URL set — training from committed GDSC data ...")
    train_main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
