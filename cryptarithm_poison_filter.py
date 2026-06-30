"""cryptarithm_poison_filter.py — audit + filter wrong-CoT training targets.

corpus.py bakes the reasoning file's OWN final \\boxed{} into the training
target (not the gold). When huikang's solver failed (e.g. the cryptarithm
stub that defaults to concatenation), that boxed answer is WRONG, so the model
is trained to produce a wrong answer with a confident trace.

This script, run AFTER corpus.py and BEFORE pack_corpus.py:
  1. AUDITS every entry that has a reasoning file: does its boxed answer match
     the gold (train.csv) under the competition verify()? Prints per-category
     poison rates. (Augmentation entries have no reasoning file and are left
     untouched.)
  2. FILTERS: sets included=False on poisoned rows in the chosen categories so
     pack_corpus.py skips them. Defaults to cryptarithm only (clean single
     variable); use --categories to widen, or --audit-only to change nothing.

Non-destructive: backs up corpus.jsonl -> corpus.jsonl.prefilter.

Usage (from repo root):
  python3 cryptarithm_poison_filter.py --audit-only          # just look
  python3 cryptarithm_poison_filter.py                        # filter cryptarithm
  python3 cryptarithm_poison_filter.py --categories all       # filter every poisoned row
  python3 pack_corpus.py
  modal volume put nemotron ./corpus_packed.jsonl /corpus-v4/corpus.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
from collections import defaultdict
from pathlib import Path

MANIFEST = Path("corpus.jsonl")
TRAIN_CSV = Path("train.csv")
REASONING_DIR = Path("reasoning")
CRYPT_DEFAULT = {"cryptarithm_deduce", "cryptarithm_guess"}


def last_boxed(text: str) -> str | None:
    m = re.findall(r"\\boxed\{([^}]*)\}", text)
    return m[-1].strip() if m else None


def verify(gold: str, pred: str | None) -> bool:
    """Mirror the competition metric: binary->strict, numeric->isclose, else ci-string."""
    if pred is None:
        return False
    g, p = gold.strip(), pred.strip()
    if re.fullmatch(r"[01]+", g):
        return g == p
    try:
        return math.isclose(float(g), float(p), rel_tol=1e-2, abs_tol=1e-5)
    except (ValueError, TypeError):
        pass
    return g.lower() == p.lower()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit-only", action="store_true")
    ap.add_argument("--categories", default="cryptarithm",
                    help="'cryptarithm' (default), 'all', or comma-separated category names")
    args = ap.parse_args()

    gold = {r["id"]: r["answer"].strip()
            for r in csv.DictReader(open(TRAIN_CSV, newline=""))}
    rows = [json.loads(l) for l in open(MANIFEST) if l.strip()]

    # AUDIT every reasoning-backed entry
    audit: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])  # [ok, poison, no_reasoning]
    judged: dict[str, bool] = {}
    for rec in rows:
        cat, pid = rec.get("category"), rec["problem_id"]
        fp = REASONING_DIR / f"{pid}.txt"
        if not fp.is_file():
            audit[cat][2] += 1               # augmentation / no solver CoT — untouched
            continue
        ok = pid in gold and verify(gold[pid], last_boxed(fp.read_text()))
        judged[pid] = ok
        audit[cat][0 if ok else 1] += 1

    print(f"{'category':<26}{'ok':>7}{'poison':>8}{'poison%':>9}{'no-CoT':>8}")
    for cat in sorted(audit):
        ok, poison, none = audit[cat]
        tot = ok + poison
        rate = f"{100*poison/tot:.0f}%" if tot else "-"
        print(f"{cat:<26}{ok:>7}{poison:>8}{rate:>9}{none:>8}")

    if args.audit_only:
        print("\n--audit-only: no changes written.")
        return

    if args.categories == "cryptarithm":
        target = CRYPT_DEFAULT
    elif args.categories == "all":
        target = {rec.get("category") for rec in rows}
    else:
        target = set(args.categories.split(","))

    shutil.copy(MANIFEST, str(MANIFEST) + ".prefilter")
    dropped = 0
    for rec in rows:
        pid = rec["problem_id"]
        if rec.get("category") in target and pid in judged and not judged[pid]:
            rec["included"] = False
            dropped += 1
    with open(MANIFEST, "w") as f:
        for rec in rows:
            f.write(json.dumps(rec) + "\n")

    print(f"\nfiltered categories: {sorted(target)}")
    print(f"rows set included=False: {dropped}")
    print(f"backup: {MANIFEST}.prefilter")
    print("next: python3 pack_corpus.py  ->  upload to /corpus-v4/corpus.jsonl")


if __name__ == "__main__":
    main()
