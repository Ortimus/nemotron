"""build_eval_set.py — assemble a HELD-OUT per-category eval set for modal_eval_v32.py,
with the integrity checks that decide whether a local eval is even meaningful.

The one check that matters: every problem written to eval_set.jsonl must NOT be in
the corpus v32 trained on. This script enforces that and, if you don't hand it a
holdout list, DERIVES the untrained set for you (train.csv minus trained ids) — so
its output also answers "do I even have a clean holdout?".

Sources (all local, from your repo root):
  --train          train.csv                (id, prompt, answer)
  --manifest       corpus.jsonl             (id -> category; included flag)
  --trained-corpus corpus_packed.jsonl      (the file v32 actually trained on;
                                              its problem_ids = the trained set.
                                              falls back to manifest included=True)
  --holdout-file   (optional) ids one per line you reserved; if omitted, derive

Output: eval_set.jsonl  ({"id","prompt","answer","category"}, prompt = bare puzzle).

Usage:
  python3 build_eval_set.py                                  # derive holdout
  python3 build_eval_set.py --holdout-file holdout_ids.txt   # use a reserved split
  python3 build_eval_set.py --max-per-cat 80                 # subsample for speed
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

SUFFIX_MARKER = "Please put your final answer inside"


def load_jsonl(path: Path):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="train.csv")
    ap.add_argument("--manifest", default="corpus.jsonl")
    ap.add_argument("--trained-corpus", default="corpus_packed.jsonl")
    ap.add_argument("--holdout-file", default=None)
    ap.add_argument("--out", default="eval_set.jsonl")
    ap.add_argument("--min-per-cat", type=int, default=10)
    ap.add_argument("--max-per-cat", type=int, default=0, help="0 = no cap")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    for p in (args.train, args.manifest):
        if not Path(p).is_file():
            sys.exit(f"ERROR: required file not found: {p} (run from repo root)")

    # train.csv: prompt + answer
    tr = {r["id"]: r for r in csv.DictReader(open(args.train, newline=""))}

    # manifest: id -> category, and (fallback) trained ids
    id2cat: dict[str, str] = {}
    manifest_trained: set[str] = set()
    for rec in load_jsonl(Path(args.manifest)):
        id2cat[rec["problem_id"]] = rec.get("category", "unknown")
        if rec.get("included", True):
            manifest_trained.add(rec["problem_id"])

    # trained ids: prefer the actual packed training file
    if Path(args.trained_corpus).is_file():
        trained = {rec["problem_id"] for rec in load_jsonl(Path(args.trained_corpus))}
        print(f"Trained ids: {len(trained)} (from {args.trained_corpus})")
    else:
        trained = manifest_trained
        print(f"Trained ids: {len(trained)} (from {args.manifest} included=True; "
              f"{args.trained_corpus} not found)")

    # candidate holdout ids
    if args.holdout_file:
        if not Path(args.holdout_file).is_file():
            sys.exit(f"ERROR: --holdout-file not found: {args.holdout_file}")
        cands = [x.strip() for x in open(args.holdout_file) if x.strip()]
        print(f"Candidate holdout: {len(cands)} ids (from {args.holdout_file})")
    else:
        cands = [i for i in tr if i not in trained]
        print(f"Candidate holdout: {len(cands)} ids (DERIVED = train.csv - trained)")

    # ── CHECK 1: contamination (the one that matters) ──
    contaminated = [i for i in cands if i in trained]
    if contaminated:
        print(f"\n⚠️  CONTAMINATION: {len(contaminated)} candidate ids ARE in the "
              f"training set — these were seen by v32 and are NOT held out.")
        print(f"    sample: {contaminated[:8]}")
        cands = [i for i in cands if i not in trained]
        print(f"    dropped them; {len(cands)} clean candidates remain.")

    # ── CHECK 2: presence (have prompt/answer/category) ──
    missing_train = [i for i in cands if i not in tr]
    missing_cat = [i for i in cands if i not in id2cat]
    if missing_train:
        print(f"⚠️  {len(missing_train)} ids not in train.csv (no prompt/answer); dropped.")
    if missing_cat:
        print(f"⚠️  {len(missing_cat)} ids not in manifest (no category); dropped.")
    cands = [i for i in cands if i in tr and i in id2cat]

    # ── CHECK 3: sanity (nonempty, no pre-applied suffix, dedup) ──
    seen: set[str] = set()
    clean: list[str] = []
    bad_empty = bad_suffix = dup = 0
    for i in cands:
        if i in seen:
            dup += 1
            continue
        seen.add(i)
        prompt, ans = tr[i]["prompt"], tr[i]["answer"]
        if not prompt.strip() or not str(ans).strip():
            bad_empty += 1
            continue
        if SUFFIX_MARKER in prompt:
            bad_suffix += 1            # prompt already has the boxed suffix; eval adds it -> double
            continue
        clean.append(i)
    if dup:        print(f"⚠️  {dup} duplicate ids dropped.")
    if bad_empty:  print(f"⚠️  {bad_empty} empty prompt/answer dropped.")
    if bad_suffix: print(f"⚠️  {bad_suffix} prompts already contain the boxed suffix; "
                         f"dropped (eval adds it -> would double).")

    if not clean:
        print("\n❌ No clean held-out problems. v32 appears to have trained on every "
              "train.csv id, so a local per-category eval would measure memorization, "
              "not generalization. Skip it: trust the audit + the LB. (See notes.)")
        return

    # ── CHECK 4: per-category balance + optional subsample ──
    by_cat: dict[str, list[str]] = defaultdict(list)
    for i in clean:
        by_cat[id2cat[i]].append(i)

    rng = random.Random(args.seed)
    selected: list[str] = []
    for cat, ids in by_cat.items():
        if args.max_per_cat and len(ids) > args.max_per_cat:
            ids = rng.sample(ids, args.max_per_cat)
        selected.extend(ids)

    final_counts = Counter(id2cat[i] for i in selected)
    thin = [c for c, n in final_counts.items() if n < args.min_per_cat]

    # ── write ──
    with open(args.out, "w") as f:
        for i in selected:
            f.write(json.dumps({
                "id": i,
                "prompt": tr[i]["prompt"],
                "answer": str(tr[i]["answer"]).strip(),
                "category": id2cat[i],
            }) + "\n")

    print(f"\nWrote {len(selected)} problems to {args.out}")
    print(f"{'category':<26}{'n':>6}")
    for cat in sorted(final_counts):
        print(f"{cat:<26}{final_counts[cat]:>6}")
    if thin:
        print(f"\n⚠️  thin categories (< {args.min_per_cat}, noisy accuracy): "
              f"{', '.join(sorted(thin))}")
    print(f"\nNext:  modal volume put nemotron ./{args.out} /eval/eval_set.jsonl"
          f"  &&  modal run modal_eval_v32.py")


if __name__ == "__main__":
    main()
