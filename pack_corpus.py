#!/usr/bin/env python3
"""
pack_corpus.py — build the tokens-inline jsonl the Modal loader expects, by
reconstructing flat tokens+mask from huikang's per-problem SEGMENT files.

His corpus.py writes:
  - corpus.jsonl                       (metadata index: problem_id, included, ...)
  - corpus/<problem_id>/synthetic.jsonl  (INTERLEAVED SEGMENTS, not flat tokens)

Each segment line is {"type": "masked"|"unmasked", "pos": int, "tokens": [...]}.
We concatenate segments in order to rebuild the full token sequence, and set
mask=1 for 'unmasked' segment tokens, 0 for 'masked'. This reverses build_segments().

Output: corpus_packed.jsonl with {problem_id, tokens, mask} per line — the format
train_remote() reads.

Usage:
  python3 pack_corpus.py
  modal volume rm  nemotron /corpus-v1/corpus.jsonl
  modal volume put nemotron ./corpus_packed.jsonl /corpus-v1/corpus.jsonl
"""
import json
import os
from pathlib import Path

MANIFEST = Path("corpus.jsonl")
CORPUS_DIR = Path("corpus")          # per-problem dirs live here: corpus/<id>/synthetic.jsonl
OUT = Path("corpus_packed.jsonl")


def reconstruct(seg_path: Path):
    """Read segment lines, return (tokens, mask) by concatenating in file order.
    Segments are written in positional order by build_segments, so file order == seq order.
    Sort by 'pos' defensively in case ordering isn't guaranteed."""
    segs = []
    with open(seg_path) as f:
        for line in f:
            line = line.strip()
            if line:
                segs.append(json.loads(line))
    segs.sort(key=lambda s: s.get("pos", 0))
    tokens, mask = [], []
    for s in segs:
        t = s["tokens"]
        tokens.extend(t)
        mask.extend([1] * len(t) if s["type"] == "unmasked" else [0] * len(t))
    return tokens, mask


def main():
    if not MANIFEST.is_file():
        raise SystemExit(f"Manifest not found: {MANIFEST} (run from repo root)")
    if not CORPUS_DIR.is_dir():
        # corpus/ may be elsewhere; try to locate it
        cands = list(Path(".").glob("**/corpus/*/synthetic.jsonl"))
        if cands:
            base = cands[0].parent.parent
            print(f"Using corpus dir: {base}")
        else:
            raise SystemExit("Could not find corpus/<id>/synthetic.jsonl tree.")
    else:
        base = CORPUS_DIR

    n_written = n_excluded = n_missing = n_empty = 0
    with open(MANIFEST) as mf, open(OUT, "w") as out:
        for line in mf:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if not rec.get("included", True):
                n_excluded += 1
                continue
            pid = rec["problem_id"]
            seg_path = base / pid / "synthetic.jsonl"
            if not seg_path.is_file():
                n_missing += 1
                if n_missing <= 5:
                    print(f"  missing: {seg_path}")
                continue
            tokens, mask = reconstruct(seg_path)
            if not tokens or not any(mask):
                n_empty += 1
                continue
            # sanity: counts should match the manifest
            exp_tok = rec.get("token_count")
            if exp_tok is not None and len(tokens) != exp_tok:
                print(f"  WARN {pid}: token_count {exp_tok} != reconstructed {len(tokens)}")
            out.write(json.dumps({"problem_id": pid, "tokens": tokens, "mask": mask}) + "\n")
            n_written += 1

    print(f"Wrote {n_written} entries to {OUT}")
    print(f"  excluded (included=False): {n_excluded}")
    print(f"  missing segment files:     {n_missing}")
    print(f"  empty/all-masked skipped:  {n_empty}")
    if n_written:
        sz = OUT.stat().st_size / 1e6
        print(f"  {OUT} size: {sz:.1f} MB")


if __name__ == "__main__":
    main()
