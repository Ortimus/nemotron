"""Extract equation_numeric_guess failures for analysis.
Run from the nemotron repo root after reasoning.py has run.
Outputs eng_failures.json with the unsolved problems + their examples.
"""
import json
from pathlib import Path

problems_index = Path("problems.jsonl")
entries = {}
with open(problems_index) as f:
    for line in f:
        if line.strip():
            e = json.loads(line)
            entries[e["id"]] = e

# equation_numeric_guess problems that are NOT rule_found
targets = []
for pid, e in entries.items():
    if e["category"] == "equation_numeric_guess" and e.get("status") != "rule_found":
        # load the full problem (examples + question + answer)
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

print(f"equation_numeric_guess failures: {len(targets)}")
with open("eng_failures.json", "w") as f:
    json.dump(targets, f, indent=2)
print("Wrote eng_failures.json")

# Also dump a few inline for quick view
for t in targets[:5]:
    print(f"\nID {t['id']}: Q={t['question']} A={t['answer']}")
    for inp, out in t["examples"]:
        print(f"  {inp} = {out}")
