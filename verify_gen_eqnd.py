"""Independently verify a gen_eqnd/ batch before merging into the corpus.

Re-derives correctness from scratch (does not trust the generator):
  1. the solver reproduces the stored answer,
  2. the query operator is covered by the examples (true deduce),
  3. the reasoning FILE's last \\boxed{} equals the answer
     (this is what corpus.py bakes into the training target),
  4. every id is present in gen_problems.jsonl,
  5. prompts are unique,
  6. CoTs fit the 8192 token budget.

Run from repo root:  python verify_gen_eqnd.py --dir gen_eqnd
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re

from reasoners.equation_numeric import reasoning_equation_numeric
from reasoners.store_types import Problem, Example


def boxed(s: str | None) -> str | None:
    m = re.findall(r"\\boxed\{([^}]*)\}", s or "")
    return m[-1].strip() if m else None


def parse_prompt(p: str):
    exs = []
    for ln in p.splitlines():
        if "=" in ln and "determine" not in ln.lower():
            L, R = ln.split("=", 1)
            exs.append(Example(L.strip(), R.strip()))
    q = p.split("determine the result for:")[-1].strip()
    return exs, q


def op_of(seg: str) -> str | None:
    m = re.match(r"^\s*\d+(\D)\d+", seg)
    return m.group(1) if m else None


def main(d: str) -> None:
    rows = list(csv.DictReader(open(os.path.join(d, "gen_train.csv"), newline="")))
    probs = {json.loads(l)["id"] for l in open(os.path.join(d, "gen_problems.jsonl"))}

    n = len(rows)
    solver_ok = cov = file_ok = in_index = 0
    lens: list[int] = []
    seen: set[str] = set()
    dup = 0
    bad: list[str] = []

    for r in rows:
        pid, gold = r["id"], r["answer"].strip()
        exs, q = parse_prompt(r["prompt"])

        t = reasoning_equation_numeric(
            Problem(id=pid, category="equation_numeric_deduce",
                    examples=exs, question=q, answer=gold)
        )
        if boxed(t) == gold:
            solver_ok += 1
        else:
            bad.append(pid)

        exops = {op_of(str(e.input_value)) for e in exs}
        if op_of(q) in exops:
            cov += 1

        fp = os.path.join(d, "reasoning", f"{pid}.txt")
        if os.path.isfile(fp):
            txt = open(fp).read()
            lens.append(len(txt))
            if boxed(txt) == gold:
                file_ok += 1

        if pid in probs:
            in_index += 1
        if r["prompt"] in seen:
            dup += 1
        seen.add(r["prompt"])

    lens.sort()
    print(f"records:                     {n}")
    print(f"solver reproduces answer:    {solver_ok}/{n}")
    print(f"deduce coverage (query op):  {cov}/{n}")
    print(f"reasoning-file boxed==gold:  {file_ok}/{n}")
    print(f"ids in gen_problems.jsonl:   {in_index}/{n}")
    print(f"unique prompts:              {n - dup}/{n}")
    if lens:
        print(f"CoT chars: median={lens[len(lens) // 2]} max={lens[-1]} "
              f"(~{lens[-1] // 4} tok, limit 8192)")
    if bad:
        print(f"FAIL ids (first 10): {bad[:10]}")
    clean = solver_ok == n and cov == n and file_ok == n and in_index == n and dup == 0
    print("ALL GOOD ✅" if clean else "ISSUES FOUND ❌")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="gen_eqnd")
    args = ap.parse_args()
    main(args.dir)
