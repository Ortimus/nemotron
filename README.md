# NVIDIA Progress Prize submission

This is the Github repository to the Progress Prize winning submission for NVIDIA Nemotron Model Reasoning Challenge.

Resources on Kaggle

- [Writeup](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/689915).
- [Notebook](https://www.kaggle.com/code/huikang/end-to-end-finetuning-for-lb-0-85)


## Tabs on nemotron.huikang.dev

- **[Base](https://nemotron.huikang.dev/base.html)** — Grid of competition problems grouped by category, color-coded as solved / partially solved / unsolved (combining the labeled status with per-run generation correctness). Click a problem for its prompt, parsed transformation table, answer, per-run extracted answer, and the token-level generation trace colored by logprob.
- **[Synthetic](https://nemotron.huikang.dev/synthetic.html)** — Grid of synthetic problem examples grouped by category, color-coded by investigation status (rule found / hypothesis formed / rule unknown). Click a problem for its prompt, parsed transformation, answer, reasoning, and investigation notes.
- **[Corpus](https://nemotron.huikang.dev/corpus.html)** — Sortable table of training corpus entries with masked, unmasked, and total token counts per row. Filter by category or problem ID; open a row to see the token-level trace with masking highlighted.
- **[Training](https://nemotron.huikang.dev/training.html)** — Per-problem table of step, loss-token count, and minimum logprob across training epochs. Select an epoch and a row to see token-level logprob changes against the base model.
- **[Metrics](https://nemotron.huikang.dev/metrics.html)** — Multi-run comparison with charts for loss per token (overall and by category), min logprob by category, gradient norm, learning rate, and step time. Cmd+click a legend entry to isolate that category.


Running the webpage locally

```sh
./serve.sh
```

Serves the static site at `http://localhost:33304/`.


## Executing training


```
uv run python3 reasoning.py
uv run python3 augmentation.py
uv run python3 corpus.py
uv run python3 train_sft.py
uv run modal run upload_adapter.py
```
