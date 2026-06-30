"""Extract bit_manipulation failures for analysis.
Run from the nemotron repo root after reasoning.py has run.
"""
import json
from pathlib import Path

entries = {}
with open("problems.jsonl") as f:
    for line in f:
        if line.strip():
            e = json.loads(line)
            entries[e["id"]] = e

targets = []
solved = 0
for pid, e in entries.items():
    if e["category"] != "bit_manipulation":
        continue
    if e.get("status") == "rule_found":
        solved += 1
        continue
    prob_path = Path("problems") / f"{pid}.jsonl"
    if prob_path.exists():
        with open(prob_path) as pf:
            prob = json.loads(pf.readline())
        targets.append({
            "id": pid,
            "examples": [(ex["input_value"], ex["output_value"]) for ex in prob["examples"]],
            "question": prob["question"],
            "answer": prob["answer"],
        })

print(f"bit_manipulation: {solved} solved, {len(targets)} failures")
with open("bitman_failures.json", "w") as f:
    json.dump(targets, f, indent=2)
print("Wrote bitman_failures.json")

for t in targets[:3]:
    print(f"\nID {t['id']}: Q={t['question']} A={t['answer']}")
    for inp, out in t["examples"][:4]:
        print(f"  {inp} -> {out}")
    print(f"  ...({len(t['examples'])} examples total)")
