"""
generate_bitmanip_data.py — build clean, verified SFT data for bit_manipulation.

Two sources, both guaranteed-correct:

1. SOLVED TRAIN: run the deterministic solver over the real train puzzles; keep a
   record only when the solver recovers a rule AND its answer matches the gold
   answer (double verification). ~934 records.

2. FORWARD-GENERATED: sample a rule from the EMPIRICAL distribution the solver
   found in real puzzles, generate fresh random examples + a query, compute the
   answer forward (trivially correct), build the puzzle in the exact competition
   prompt format, then SELF-VERIFY: re-solve from the examples alone and keep it
   only if the recovered answer matches and the puzzle is unambiguous. This is the
   eqnd self-verifying-generator trick — every emitted example is provably correct
   and well-posed.

Output: bitmanip_sft.jsonl, records of {id, category, source, prompt, completion, answer}.
The `prompt` carries the eval boxed-instruction; `completion` is the <think>..</think>
\\boxed{} CoT. Tokenize + completion-mask these with apply_chat_template in your Modal
pipeline (prompt tokens mask=0, completion tokens mask=1) and append to corpus_packed.jsonl.

Usage: python3 generate_bitmanip_data.py [N_FORWARD]
"""
from __future__ import annotations
import csv, json, random, hashlib, os, argparse

from bit_manipulation_solver import parse_prompt, recover_rule, solve_prompt

# Data dir defaults to $NEMOTRON_DATA (falls back to the container path); all
# paths are overridable on the CLI. Locally: export NEMOTRON_DATA=/path/to/data
# or pass --train/--corpus/--out explicitly.
DATA_DIR = os.environ.get("NEMOTRON_DATA", "/mnt/user-data/uploads")

BOXED = ("\nPlease put your final answer inside `\\boxed{}`. "
         "For example: `\\boxed{your answer}`")
HEADER = ("In Alice's Wonderland, a secret bit manipulation rule transforms 8-bit "
          "binary numbers. The transformation involves operations like bit shifts, "
          "rotations, XOR, AND, OR, NOT, and possibly majority or choice functions.\n\n"
          "Here are some examples of input -> output:\n")


def build_prompt(exs, q):
    body = "".join(f"{a} -> {b}\n" for a, b in exs)
    return f"{HEADER}{body}\nNow, determine the output for: {q}"


def main(train_csv, corpus, out, n_forward=4000):
    tr = {r['id']: r for r in csv.DictReader(open(train_csv, newline=''))}
    cat = {}
    for l in open(corpus):
        r = json.loads(l); cat[r['problem_id']] = r['category']
    bm = [pid for pid, c in cat.items() if c == 'bit_manipulation' and pid in tr]

    records = []
    rule_pool = []          # (func, mask) recovered from real puzzles -> empirical distribution
    seen_prompts = set()

    # ---- 1. solved real train puzzles (both stages) ----
    solved = wrong = s1 = s2 = 0
    for pid in bm:
        prompt = tr[pid]['prompt'].strip()
        exs, q = parse_prompt(prompt)
        if not q or len(exs) < 2:
            continue
        ans, cot = solve_prompt(prompt)
        if ans is None:
            continue
        if ans != tr[pid]['answer'].strip():
            wrong += 1
            continue                      # solver disagreed with gold -> drop
        solved += 1
        rule = recover_rule(exs)          # stage-1 rule -> seed forward-gen pool
        if rule is not None:
            _, _, F, mask = rule
            rule_pool.append((F, mask)); s1 += 1
        else:
            s2 += 1
        seen_prompts.add(hashlib.md5(prompt.encode()).hexdigest())
        records.append({"id": pid, "category": "bit_manipulation",
                        "source": "solver_train",
                        "prompt": prompt + BOXED, "completion": cot, "answer": ans})
    print(f"[1] solved train: {solved} verified records "
          f"({s1} op-family + {s2} per-bit; {wrong} solver!=gold dropped)")

    # ---- 2. forward-generated, self-verified ----
    rng = random.Random(0)
    gen = attempts = 0
    while gen < n_forward and attempts < n_forward * 40:
        attempts += 1
        F, mask = rng.choice(rule_pool)
        n_ex = rng.choice([7, 8, 9])
        vals = rng.sample(range(256), n_ex + 1)         # distinct inputs + 1 query
        ins, qv = vals[:n_ex], vals[n_ex]
        exs = [(format(i, '08b'), format(F(i) ^ mask, '08b')) for i in ins]
        q = format(qv, '08b')
        prompt = build_prompt(exs, q)
        h = hashlib.md5(prompt.encode()).hexdigest()
        if h in seen_prompts:
            continue
        forward_ans = format(F(qv) ^ mask, '08b')
        # SELF-VERIFY from examples alone
        ans, cot = solve_prompt(prompt)
        if ans is None or ans != forward_ans:
            continue                       # ambiguous / not recoverable -> drop
        seen_prompts.add(h)
        gen += 1
        records.append({"id": f"bmgen_{gen:06d}", "category": "bit_manipulation",
                        "source": "forward_gen",
                        "prompt": prompt + BOXED, "completion": cot, "answer": ans})
    print(f"[2] forward-generated: {gen} self-verified records "
          f"({attempts} attempts, {100*gen/max(attempts,1):.0f}% well-posed)")

    with open(out, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"\nTOTAL clean verified bit_manipulation SFT records: {len(records)}")
    print(f"Written to {out}")

    # samples
    for src in ("solver_train", "forward_gen"):
        ex = next(r for r in records if r["source"] == src)
        print("\n" + "=" * 70 + f"\n=== sample ({src})  id={ex['id']}  answer={ex['answer']} ===")
        print("USER:", ex["prompt"][:200].replace("\n", " ") + " ...")
        print("ASSISTANT:\n" + ex["completion"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generate clean, verified bit_manipulation SFT data.")
    ap.add_argument("--train",     default=os.path.join(DATA_DIR, "train.csv"))
    ap.add_argument("--corpus",    default=os.path.join(DATA_DIR, "corpus.jsonl"))
    ap.add_argument("--out",       default="bitmanip_sft.jsonl")
    ap.add_argument("--n-forward", type=int, default=4000,
                    help="number of forward-generated (augmentation) records")
    a = ap.parse_args()
    main(a.train, a.corpus, a.out, a.n_forward)
