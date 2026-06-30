"""Generate synthetic equation_numeric_deduce problems with correct worked CoTs.

Approach (self-verifying):
  1. Build a puzzle by assigning each operator symbol a numeric rule taken from
     the solver's OWN candidate library, so outputs are by-definition consistent
     with what reasoning_equation_numeric() will compute.
  2. Run the real solver on the constructed problem.
  3. KEEP the sample only if the solver recovers the exact answer with a
     confident trace (no "unknown"/"default" fallback). This guarantees:
       - the CoT is correct (boxed == gold),
       - the rule is uniquely determined by the examples (true "deduce"),
       - the query operator is covered (we only query a shown operator).

Outputs (drop-in for huikang's corpus.py):
  <out>/reasoning/<id>.txt        worked CoT, ending in \\boxed{answer}
  <out>/gen_train.csv             id,prompt,answer   (append to train.csv)
  <out>/gen_problems.jsonl        {"id","category"}  (append to problems.jsonl)

Run from the repo root (so `reasoners/` is importable):
    python gen_equation_numeric_deduce.py --n 3000 --seed 0 --out gen_eqnd
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re

from reasoners.equation_numeric import reasoning_equation_numeric, _all_candidates
from reasoners.store_types import Problem, Example

PROMPT_HEADER = (
    "In Alice's Wonderland, a secret set of transformation rules is applied "
    "to equations. Below are a few examples:"
)
PROMPT_TAIL = "Now, determine the result for: "

# Operator alphabet observed in the real equation-family prompts.
OP_SYMBOLS = list("*+#}|\"/\\`>[{'?@]!)^$&(:%<-")

# Subset of the solver's library that yields realistic short outputs.
# (Names must match keys returned by _all_candidates.)
OP_POOL = [
    "addition",
    "subtraction (a-b)",
    "reverse subtraction (b-a)",
    "absolute difference",
    "negated absolute difference",
    "concatenation",
    "reverse concatenation",
    "multiplication",
    "digit absolute diff",
    "digit add mod10",
    "digit sub mod10",
    "integer division (a/b)",
    "modulo (a mod b)",
    "determinant",
    "abs determinant",
]


def out_for(op_name: str, a: int, b: int) -> str | None:
    """Output for op_name on (a, b), computed by the solver's own candidates."""
    d = dict(_all_candidates(a, b, str(a), str(b)))
    return d.get(op_name)


def _rand_operands(rng: random.Random) -> tuple[int, int]:
    # 2-digit operands -> valid for digit-wise ops and matches real distribution
    return rng.randint(10, 99), rng.randint(10, 99)


def make_puzzle(rng: random.Random):
    """Construct one candidate puzzle. Returns (prompt, examples, question, gold)."""
    k = rng.choice([2, 2, 2, 3])  # mostly 2 operators (4-5 examples), sometimes 3
    syms = rng.sample(OP_SYMBOLS, k)
    ops = rng.sample(OP_POOL, k)
    sym2op = dict(zip(syms, ops))

    # Every operator appears at least twice so the solver can uniquely determine
    # its rule, and n_ex >= 2*k guarantees the query operator is always covered.
    n_ex = rng.choice([2 * k, 2 * k + 1])
    order: list[str] = [s for s in syms for _ in range(2)]
    while len(order) < n_ex:
        order.append(rng.choice(syms))
    rng.shuffle(order)

    lines: list[str] = []
    exs: list[Example] = []
    for sym in order:
        op = sym2op[sym]
        for _ in range(200):
            a, b = _rand_operands(rng)
            o = out_for(op, a, b)
            if o not in (None, ""):
                break
        else:
            return None
        lines.append(f"{a}{sym}{b} = {o}")
        exs.append(Example(f"{a}{sym}{b}", str(o)))

    qsym = rng.choice(syms)
    qop = sym2op[qsym]
    for _ in range(200):
        qa, qb = _rand_operands(rng)
        gold = out_for(qop, qa, qb)
        if gold not in (None, ""):
            break
    else:
        return None
    question = f"{qa}{qsym}{qb}"
    prompt = PROMPT_HEADER + "\n" + "\n".join(lines) + "\n" + PROMPT_TAIL + question
    return prompt, exs, question, str(gold)


def _boxed(s: str | None) -> str | None:
    m = re.findall(r"\\boxed\{([^}]*)\}", s or "")
    return m[-1].strip() if m else None


def generate(n_keep: int, seed: int, out_dir: str) -> None:
    rng = random.Random(seed)
    reasoning_dir = os.path.join(out_dir, "reasoning")
    os.makedirs(reasoning_dir, exist_ok=True)

    kept: list[dict] = []
    tried = 0
    rejected_lowconf = 0
    rejected_mismatch = 0

    while len(kept) < n_keep:
        tried += 1
        if tried > n_keep * 50:
            print(f"WARNING: stopping early, only kept {len(kept)}/{n_keep}")
            break
        built = make_puzzle(rng)
        if built is None:
            continue
        prompt, exs, question, gold = built
        prob = Problem(
            id="x",
            category="equation_numeric_deduce",
            examples=exs,
            question=question,
            answer=gold,
        )
        trace = reasoning_equation_numeric(prob)
        if trace is None:
            continue
        low = trace.lower()
        if "unknown" in low or "default" in low:
            rejected_lowconf += 1
            continue
        if _boxed(trace) != gold:
            rejected_mismatch += 1
            continue

        pid = f"gen_eqnd_{len(kept):06d}"
        with open(os.path.join(reasoning_dir, f"{pid}.txt"), "w") as f:
            f.write(trace.rstrip("\n") + "\n")
        kept.append(
            {"id": pid, "category": "equation_numeric_deduce",
             "prompt": prompt, "answer": gold}
        )

    with open(os.path.join(out_dir, "gen_train.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "prompt", "answer"])
        w.writeheader()
        for r in kept:
            w.writerow({"id": r["id"], "prompt": r["prompt"], "answer": r["answer"]})

    with open(os.path.join(out_dir, "gen_problems.jsonl"), "w") as f:
        for r in kept:
            f.write(json.dumps({"id": r["id"], "category": r["category"]}) + "\n")

    keep_rate = 100 * len(kept) / tried if tried else 0
    print(f"kept {len(kept)} / tried {tried}  ({keep_rate:.0f}% keep rate)")
    print(f"  rejected (low-confidence/non-deduce): {rejected_lowconf}")
    print(f"  rejected (solver answer != gold):     {rejected_mismatch}")
    print(f"  reasoning/  ->  {reasoning_dir}/*.txt")
    print(f"  train rows  ->  {out_dir}/gen_train.csv")
    print(f"  problems    ->  {out_dir}/gen_problems.jsonl")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="gen_eqnd")
    args = ap.parse_args()
    generate(args.n, args.seed, args.out)
