#!/usr/bin/env python3
"""
pack_bitmanip_modal.py — fold the net-new bit_manipulation examples from
bitmanip_sft.jsonl into a new packed corpus, tokenized + masked IDENTICALLY to
how corpus.py built corpus-v3 (so train_remote() reads them the same way).

Convention (from your corpus.py):
  prompt  -> apply_chat_template([{user: prompt}], add_generation_prompt=True,
             enable_thinking=True)   # the chat template supplies the opening <think>\\n
  completion (corpus.py) = reasoning + "\\n</think>\\n\\boxed{ans}<|im_end|>"
  Your bitmanip completion = "<think>\\n" + reasoning + "\\n</think>\\n\\boxed{ans}"
    -> transform: strip leading "<think>\\n", append "<|im_end|>"
  tokens = prompt_ids + completion_ids ; mask = 0 on prompt, 1 on completion
  truncate to TOKEN_LIMIT (8192)

Only IDs NOT already in corpus-v3 are added (3,000 net-new); the 1,265 originals
already in v32 are left untouched -> clean one-variable add.

RUN (from the dir holding bitmanip_sft.jsonl and corpus.jsonl):
  modal run pack_bitmanip_modal.py

Then read the verification block in the logs, confirm the new vs existing
boundaries match, and launch training on /vol/corpus-v3-bm/corpus.jsonl.
"""
import modal

VOLUME_NAME = "nemotron"
BASE_MODEL  = "/vol/base-model"
SRC_CORPUS  = "/vol/corpus-v3/corpus.jsonl"        # what v32 trained on
OUT_CORPUS  = "/vol/corpus-v3-bm/corpus.jsonl"     # new = v3 + net-new bit_manip
BM_SFT      = "/data/bitmanip_sft.jsonl"
MANIFEST    = "/data/corpus_manifest.jsonl"        # id->category, for verification
TOKEN_LIMIT = 8192

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("transformers==4.57.6", "tokenizers", "sentencepiece", "protobuf", "jinja2")
    .add_local_file("bitmanip_sft.jsonl", BM_SFT)
    .add_local_file("corpus.jsonl", MANIFEST)
)

app = modal.App("pack-bitmanip", image=image)
vol = modal.Volume.from_name(VOLUME_NAME)


@app.function(volumes={"/vol": vol}, timeout=3600, memory=8192)
def pack():
    import json, os
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

    def build(ex):
        # prompt already carries the \boxed{} suffix -> use as-is
        prompt_ids = tok.apply_chat_template(
            [{"role": "user", "content": ex["prompt"]}],
            tokenize=True, add_generation_prompt=True, enable_thinking=True,
        )
        c = ex["completion"].lstrip()
        if c.startswith("<think>"):
            c = c[len("<think>"):]
            if c.startswith("\n"):
                c = c[1:]
        c = c.rstrip()
        if not c.endswith("<|im_end|>"):
            c = c + "<|im_end|>"
        comp_ids = tok.encode(c, add_special_tokens=False)
        tokens = list(prompt_ids) + list(comp_ids)
        mask = [0] * len(prompt_ids) + [1] * len(comp_ids)
        if len(tokens) > TOKEN_LIMIT:
            tokens = tokens[:TOKEN_LIMIT]
            mask = mask[:TOKEN_LIMIT]
        return tokens, mask, len(prompt_ids)

    import random
    os.makedirs(os.path.dirname(OUT_CORPUS), exist_ok=True)

    # id->category from manifest (to locate an existing bit_manip line for verification)
    cat_of = {}
    with open(MANIFEST) as mf:
        for line in mf:
            line = line.strip()
            if line:
                r = json.loads(line)
                cat_of[r["problem_id"]] = r.get("category", "?")

    # 1) read corpus-v3 verbatim into memory, collect its ids
    base_lines = []
    existing = set()
    first_bm_existing = None  # (problem_id, tokens, mask) for verification
    with open(SRC_CORPUS) as src:
        for line in src:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            existing.add(r["problem_id"])
            base_lines.append(line)
            if first_bm_existing is None and cat_of.get(r["problem_id"]) == "bit_manipulation":
                first_bm_existing = (r["problem_id"], r["tokens"], r["mask"])
    n_base = len(base_lines)

    # 2) build net-new bit_manip lines (ids not already in corpus-v3)
    new_lines = []
    n_skip = n_empty = 0
    lens = []
    first_new = None
    with open(BM_SFT) as bm:
        for line in bm:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            if ex["id"] in existing:
                n_skip += 1
                continue
            tokens, mask, n_prompt = build(ex)
            if not any(mask):
                n_empty += 1
                continue
            new_lines.append(json.dumps({"problem_id": ex["id"], "tokens": tokens, "mask": mask}))
            existing.add(ex["id"])
            lens.append(len(tokens))
            if first_new is None:
                first_new = (ex["id"], tokens, mask, n_prompt)
    n_new = len(new_lines)

    # 3) combine + SHUFFLE so the new bit_manip are distributed, not a block at the
    #    end of epoch 0 (which, with SHUFFLE_DATASET=False, is the structure we
    #    suspect behind the v36 collapse). Seeded for reproducibility.
    all_lines = base_lines + new_lines
    random.Random(0).shuffle(all_lines)
    with open(OUT_CORPUS, "w") as out:
        for l in all_lines:
            out.write(l + "\n")
    vol.commit()

    total = len(all_lines)
    print("=" * 70)
    print(f"base (corpus-v3):        {n_base}")
    print(f"net-new bit_manip added: {n_new}")
    print(f"skipped (already in v3): {n_skip}")
    print(f"empty/all-masked:        {n_empty}")
    print(f"TOTAL in new corpus:     {total}")
    if lens:
        lens.sort()
        med = lens[len(lens) // 2]
        over = sum(1 for x in lens if x >= TOKEN_LIMIT)
        print(f"new bit_manip token lens: median={med} max={max(lens)} truncated(>= {TOKEN_LIMIT})={over}")
    spe = max(total // 32, 1)
    print(f"steps/epoch={spe}  ->  for ~1.18 epochs (match v32), set NUM_STEPS={round(1.18 * spe)}")
    print(f"wrote {OUT_CORPUS}")
    print("=" * 70)

    # 3) VERIFICATION — decode boundaries; new and existing should look identical
    def show(label, tokens, mask, n_prompt=None):
        print(f"\n--- {label} ---")
        # find the prompt->completion boundary (first mask==1)
        b = next((i for i, m in enumerate(mask) if m == 1), len(mask))
        if n_prompt is not None:
            b = n_prompt
        print(f"len={len(tokens)} prompt_tokens={b} completion_tokens={len(tokens)-b}")
        print("last 24 PROMPT tokens decoded:")
        print("   " + repr(tok.decode(tokens[max(0, b - 24):b])))
        print("first 24 COMPLETION tokens decoded:")
        print("   " + repr(tok.decode(tokens[b:b + 24])))
        print("last 16 tokens decoded (should end ...</think>\\n\\boxed{..}<|im_end|>):")
        print("   " + repr(tok.decode(tokens[-16:])))
        print(f"mask around boundary: ...{mask[b-3:b]} | {mask[b:b+3]}...  (expect [0,0,0] | [1,1,1])")

    if first_new:
        pid, t, m, npr = first_new
        show(f"NEW bit_manip example  id={pid}", t, m, npr)
    if first_bm_existing:
        pid, t, m = first_bm_existing
        show(f"EXISTING corpus-v3 bit_manip  id={pid}", t, m)
    print("\nIf the NEW and EXISTING boundaries look the same shape, the packing matches. "
          "Then train on /vol/corpus-v3-bm/corpus.jsonl.")


@app.local_entrypoint()
def main():
    pack.spawn()
