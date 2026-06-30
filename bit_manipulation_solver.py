"""
bit_manipulation_solver.py — deterministic solver + CoT generator for the
Nemotron competition's bit_manipulation category.

Each puzzle gives ~8 (input -> output) examples of a hidden fixed function on
8-bit bytes plus a query. We recover the rule from a DETERMINED hypothesis space
and accept a candidate only if it reproduces EVERY example; with 8 examples x 8
bits = 64 bit-constraints, a false fit is ~2^-64, so a fit is the true rule and
generalizes. Two stages:

  STAGE 1 (whole-byte op-family): single word-ops (shifts/rotations/NOT/reverse/
  nibble-swap), their pairwise XOR/AND/OR combinations, bitwise MAJORITY over op
  triples, plus an optional trailing XOR-mask. Clean, elegant rules.

  STAGE 2 (per-bit fallback): for puzzles no whole-byte rule fits, recover each
  output bit as a small-fan-in (<=3, usually 2) Boolean function of input bits,
  determined where the query's input combination was observed.

Coverage on the full 1,602-puzzle train set: ~80% solved correct-vs-gold.

solve_prompt(prompt) -> (answer:str|None, cot:str|None)   (None,None) if unsolved.
"""
from __future__ import annotations
import re
from itertools import combinations, product

BIN = re.compile(r'^[01]{8}$')


def _rotl(x, k): return ((x << k) | (x >> (8 - k))) & 0xFF
def _rotr(x, k): return ((x >> k) | (x << (8 - k))) & 0xFF


def _base_ops():
    ops = [("id", "the input unchanged", lambda x: x),
           ("not", "the bitwise NOT", lambda x: ~x & 0xFF),
           ("rev", "the bits reversed end-to-end", lambda x: int(format(x, '08b')[::-1], 2)),
           ("swap", "the two nibbles swapped", lambda x: ((x & 0x0F) << 4) | (x >> 4))]
    for k in range(1, 8):
        ops += [(f"rotl{k}", f"a left rotation by {k}",  lambda x, k=k: _rotl(x, k)),
                (f"rotr{k}", f"a right rotation by {k}", lambda x, k=k: _rotr(x, k)),
                (f"shl{k}",  f"a left shift by {k}",     lambda x, k=k: (x << k) & 0xFF),
                (f"shr{k}",  f"a right shift by {k}",    lambda x, k=k: x >> k)]
    return ops


_BASE = _base_ops()


def _candidates():
    for n, h, f in _BASE:
        yield n, h, f
    for (n1, h1, f1), (n2, h2, f2) in product(_BASE, _BASE):
        yield f"{n1}^{n2}", f"({h1}) XOR ({h2})", lambda x, f1=f1, f2=f2: f1(x) ^ f2(x)
        yield f"{n1}&{n2}", f"({h1}) AND ({h2})", lambda x, f1=f1, f2=f2: f1(x) & f2(x)
        yield f"{n1}|{n2}", f"({h1}) OR ({h2})",  lambda x, f1=f1, f2=f2: f1(x) | f2(x)
    for (n1, h1, f1), (n2, h2, f2), (n3, h3, f3) in combinations(_BASE, 3):
        yield (f"maj({n1},{n2},{n3})",
               f"the bitwise majority of [{h1}], [{h2}], [{h3}]",
               lambda x, f1=f1, f2=f2, f3=f3: (f1(x) & f2(x)) | (f1(x) & f3(x)) | (f2(x) & f3(x)))


_CANDS = list(_candidates())


def parse_prompt(prompt):
    exs, q = [], None
    for ln in prompt.splitlines():
        ln = ln.strip()
        if '->' in ln:
            a, b = [s.strip() for s in ln.split('->')]
            if BIN.match(a) and BIN.match(b):
                exs.append((a, b))
        elif 'output for' in ln.lower():
            m = re.search(r'[01]{8}', ln)
            if m:
                q = m.group(0)
    return exs, q


def recover_rule(exs):
    """Stage 1: whole-byte op-family. Returns (label, human, func, mask) or None."""
    ins = [int(a, 2) for a, _ in exs]
    outs = [int(o, 2) for _, o in exs]
    for label, human, F in _CANDS:
        if all(F(i) == o for i, o in zip(ins, outs)):
            return label, human, F, 0
        m = outs[0] ^ F(ins[0])
        if m and all((F(i) ^ m) == o for i, o in zip(ins, outs)):
            return label, human, F, m
    return None


_GATES = {(0, 0, 0, 1): "AND", (0, 1, 1, 1): "OR", (0, 1, 1, 0): "XOR",
          (1, 1, 1, 0): "NAND", (1, 0, 0, 0): "NOR", (1, 0, 0, 1): "XNOR"}


def recover_perbit(exs, q):
    """Stage 2: each output bit as a <=3-input Boolean fn, determined at the query.
    Returns list of per-bit dicts {S, val, desc} or None."""
    ins = [int(a, 2) for a, _ in exs]
    outs = [int(o, 2) for _, o in exs]
    qv = int(q, 2)
    inb = [[(i >> b) & 1 for b in range(7, -1, -1)] for i in ins]
    outb = [[(o >> b) & 1 for b in range(7, -1, -1)] for o in outs]
    qb = [(qv >> b) & 1 for b in range(7, -1, -1)]
    spec = []
    for p in range(8):
        found = None
        for s in range(0, 4):
            for S in combinations(range(8), s):
                tbl, ok = {}, True
                for e in range(len(ins)):
                    key = tuple(inb[e][j] for j in S)
                    if key in tbl and tbl[key] != outb[e][p]:
                        ok = False; break
                    tbl[key] = outb[e][p]
                if not ok:
                    continue
                qkey = tuple(qb[j] for j in S)
                if qkey in tbl:
                    found = (S, tbl, tbl[qkey]); break
            if found is not None:
                break
        if found is None:
            return None
        S, tbl, val = found
        spec.append({"S": S, "val": val, "desc": _describe_bit(S, tbl)})
    return spec


def _fmt_entries(S, tbl):
    pos = [7 - j for j in S]
    label = "".join(f"b{p}" for p in pos)
    return label, ", ".join(f"{''.join(map(str,k))}->{tbl[k]}" for k in sorted(tbl))


def _describe_bit(S, tbl):
    pos = [7 - j for j in S]  # report as bit index from the right (LSB=0) for readability
    if len(S) == 0:
        return f"constant {next(iter(tbl.values()))}"
    if len(S) == 1:
        if tbl.get((0,)) == 0 and tbl.get((1,)) == 1:
            return f"input bit {pos[0]}"
        if tbl.get((0,)) == 1 and tbl.get((1,)) == 0:
            return f"NOT input bit {pos[0]}"
        lbl, ent = _fmt_entries(S, tbl)
        return f"input bit {pos[0]} (from examples, {lbl}: {ent})"
    if len(S) == 2:
        key = tuple(tbl.get(k) for k in [(0, 0), (0, 1), (1, 0), (1, 1)])
        if None not in key and key in _GATES:
            return f"(input bit {pos[0]} {_GATES[key]} input bit {pos[1]})"
        lbl, ent = _fmt_entries(S, tbl)
        return f"determined by input bits {pos[0]},{pos[1]} (from examples, {lbl}: {ent})"
    lbl, ent = _fmt_entries(S, tbl)
    return f"determined by input bits {','.join(map(str, pos))} (from examples, {lbl}: {ent})"


def solve_prompt(prompt):
    exs, q = parse_prompt(prompt)
    if not q or len(exs) < 2:
        return None, None
    rule = recover_rule(exs)
    if rule is not None:
        label, human, F, mask = rule
        qv = int(q, 2)
        ans = format(F(qv) ^ mask, '08b')
        return ans, _cot_opfam(exs, q, human, F, mask, ans)
    spec = recover_perbit(exs, q)
    if spec is not None:
        ans = ''.join(str(b["val"]) for b in spec)
        return ans, _cot_perbit(exs, q, spec, ans)
    return None, None


def _cot_opfam(exs, q, human, F, mask, ans):
    qv = int(q, 2)
    L = ["I need to find the fixed transformation rule mapping each 8-bit input to "
         "its output, then apply it to the query.", "",
         "Testing candidate operations (shifts, rotations, NOT, reversal, majority, "
         "and their AND/OR/XOR combinations) against the examples.",
         f"The rule reproducing every example is: output = {human}"
         + (f", then XOR with {format(mask, '08b')}" if mask else "") + ".", "",
         "Verifying on the examples:"]
    for a, o in exs[:3]:
        L.append(f"  {a} -> {format(F(int(a,2)) ^ mask, '08b')}  (matches {o})")
    L += ["  ... (holds for all examples)", "", f"Applying to the query {q}:"]
    if mask:
        L.append(f"  intermediate = {format(F(qv), '08b')}, XOR {format(mask, '08b')} = {ans}")
    else:
        L.append(f"  result = {ans}")
    return "<think>\n" + "\n".join(L) + f"\n</think>\n\\boxed{{{ans}}}"


def _cot_perbit(exs, q, spec, ans):
    qb = [(int(q, 2) >> b) & 1 for b in range(7, -1, -1)]  # b7..b0
    L = ["I need to find the transformation mapping each 8-bit input to its output.", "",
         "No single whole-byte operation fits all examples, so I'll determine each "
         "output bit from its dependence on the input bits (bit 7 is leftmost, bit 0 "
         "rightmost).", "",
         "From the examples, each output bit (written left to right, bit 7..0) is:"]
    for idx, b in enumerate(spec):
        L.append(f"  output bit {7-idx} = {b['desc']}")
    L += ["", f"The query {q} has input bits b7..b0 = {','.join(map(str, qb))}.",
          f"Looking up each output bit from the rules above gives: {ans}"]
    return "<think>\n" + "\n".join(L) + f"\n</think>\n\\boxed{{{ans}}}"


if __name__ == "__main__":
    import csv, json, time, os, argparse
    DATA_DIR = os.environ.get("NEMOTRON_DATA", "/mnt/user-data/uploads")
    ap = argparse.ArgumentParser(description="Self-test the solver on the train set.")
    ap.add_argument("--train",  default=os.path.join(DATA_DIR, "train.csv"))
    ap.add_argument("--corpus", default=os.path.join(DATA_DIR, "corpus.jsonl"))
    a = ap.parse_args()
    tr = {r['id']: r for r in csv.DictReader(open(a.train, newline=''))}
    cat = {}
    for l in open(a.corpus):
        r = json.loads(l); cat[r['problem_id']] = r['category']
    bm = [pid for pid, c in cat.items() if c == 'bit_manipulation' and pid in tr]
    s1 = s2 = correct = tried = 0
    samples = {}
    t0 = time.time()
    for pid in bm:
        tried += 1
        exs, q = parse_prompt(tr[pid]['prompt'])
        if not q or len(exs) < 2:
            continue
        stage = 1 if recover_rule(exs) is not None else 2
        ans, cot = solve_prompt(tr[pid]['prompt'])
        if ans is None:
            continue
        if stage == 1: s1 += 1
        else: s2 += 1
        if ans == tr[pid]['answer'].strip():
            correct += 1
            samples.setdefault(stage, (pid, cot, tr[pid]['answer'].strip()))
    print(f"full {tried}-puzzle train set in {time.time()-t0:.0f}s:")
    print(f"  stage-1 op-family emitted : {s1}")
    print(f"  stage-2 per-bit emitted   : {s2}")
    print(f"  CORRECT vs gold (kept)    : {correct}/{tried} ({100*correct/tried:.0f}%)")
    for st in (1, 2):
        pid, cot, gold = samples[st]
        print(f"\n=== stage-{st} sample  [{pid}] gold={gold} ===\n{cot}")
