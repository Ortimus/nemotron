# NVIDIA Progress Prize submission

This is the Github repository to the Progress Prize winning submission for NVIDIA Nemotron Model Reasoning Challenge.

The full writeup is available on [Kaggle](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/689915).

## Tabs on nemotron.huikang.dev

- **[Synthetic](https://nemotron.huikang.dev/synthetic.html)** — Interactive grid of synthetic problem examples grouped by category (cryptarithm, equation, bit manipulation, cipher, gravity, unit conversion, numeral), color-coded by status (rule found, hypothesis formed, rule unknown). Click a problem to see its prompt, parsed transformations, answer, reasoning, and investigation notes.
- **[Corpus](https://nemotron.huikang.dev/corpus.html)** — Searchable, sortable table of training corpus entries with masked/unmasked token counts per problem and category. Open a row to view the token-level trace with masking highlighted.
- **[Training](https://nemotron.huikang.dev/training.html)** — Per-problem table of training logprob data across epochs, showing loss improvements and minimum logprob values. Select an epoch to see token-level logprob changes.
- **[Metrics](https://nemotron.huikang.dev/metrics.html)** — Multi-run comparison table with line charts for loss per token, min logprob, gradient norms, learning rate, and step timing. Cmd+click legend entries to filter categories.


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
