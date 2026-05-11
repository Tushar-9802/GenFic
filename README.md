# GenFic

Parameter-efficient fine-tuning toolkit for **literary register adaptation**. One OPLoRA adapter per literary register, trained on Project Gutenberg public-domain prose, so a single base model can be conditionally steered into Victorian-formal, Romantic-ornate, Gothic-dark, plain-realist, or modernist-spare prose without losing its base instruction-following.

The architecture extends two prior papers in this repo:

- **Springer paper** — LoRA on Mistral for literary style transfer.
- **IEEE paper** — OPLoRA (orthogonal projection) + EWC against catastrophic forgetting; the §IV-D register-discrimination metrics (passive ratio, nominalization rate, lexical density, sentence length, type-token ratio) are the eval framework here.

## Why orthogonal projection

Plain LoRA on register-distinctive prose tends to overwrite the base model's instruction-following — the adapter's "default voice" comes to dominate, and the model stops obeying user briefs (length, tense, POV). OPLoRA constrains LoRA updates to lie in the orthogonal complement of the top-k singular directions of each base weight, which preserves the directions in W carrying the most pre-training information (instruction-following, factual recall) while still letting the adapter learn register. The eval question this project answers is whether OPLoRA-adapted models show measurable register shift on the §IV-D metrics *without* regression on a base instruction-following probe.

## Training format

Each chunk is presented as a Mistral instruction-tuning pair, with loss masked to the response only:

```
[INST] Continue this scene in {register} style: {brief} [/INST] {chapter_chunk}</s>
```

Briefs are synthesized offline from the chapter content using base Mistral-Instruct (1–3 sentence content-faithful summary). At training time, loss is computed only on `{chapter_chunk}</s>`; the prompt prefix is masked with `LOSS_IGNORE_INDEX = -100`.

## Register clusters

Five clusters, defined in `src/genfic/registers.py`:

| Cluster              | Authors (PD)                                              | Style target                                              |
|----------------------|-----------------------------------------------------------|-----------------------------------------------------------|
| `victorian-formal`   | Austen, Trollope, Eliot, Gaskell                          | Periodic sentences, formal diction, free indirect style   |
| `romantic-ornate`    | Brontës, Hawthorne, Scott                                 | Long sinuous sentences, elevated diction, interiority     |
| `gothic-dark`        | Poe, Stoker, Le Fanu, M. R. James                         | Atmospheric, foreboding, archaic, ornate description      |
| `plain-realist`      | Twain, London, Crane                                      | Short sentences, concrete diction, dialogue-forward       |
| `modernist-spare`    | early Joyce (*Dubliners*), Sherwood Anderson, Lardner     | Spare, declarative, rhythm-driven, ellipsis               |

Cluster validity (separation on the §IV-D axes) is checked by `scripts/build_register_clusters.py` using metrics from `scripts/style_profile.py`.

## Pipeline

```
ingest_gutenberg.py        → source/raw/gutenberg/{author}/{work}/chapter-NNN.txt
                           → source/gutenberg_imported.jsonl

build_register_clusters.py → register_clusters.json + separation report

synthesize_briefs.py       → source/briefs.jsonl  (one brief per chapter, GPU)

build_dataset.py --register {cluster}
                           → source/{cluster}.h5  (input_ids, attention_mask, prompt_lengths)

train.py     --register {cluster}
                           → runs/{cluster}/checkpoint-XXX

generate.py  --register {cluster} --brief "..."
                           → register-shifted prose
```

## Hyperparameters (defaults)

Base: `mistralai/Mistral-7B-Instruct-v0.2`, 4-bit NF4 quantization.

LoRA: `r=32`, `alpha=64`, `dropout=0.1`, target modules q/k/v/o/gate/up/down_proj.
LoRA+: `lr_a=1e-4`, `lr_b=8e-4` (η=8×), 8-bit AdamW, cosine schedule, warmup 3 %.
OPLoRA: `k=64` — preserves enough of the top base-weight singular subspace that prompt templates still steer the model after fine-tuning.
Per-author cap: `--max-chunks-per-author 200` so no single author dominates a cluster.
Save / eval: `save_steps=250`, `eval_steps=250`; best checkpoint by `eval_loss` is reloaded at end (`load_best_model_at_end=True`).
Early stop: `--early-stop-patience 2 --early-stop-threshold 0.02` — training halts after two consecutive evals where `eval_loss` improves by less than 0.02.

## Smoke test

End-to-end on one cluster (Victorian-formal) before scaling to the rest:

1. `python scripts/ingest_gutenberg.py --cluster victorian-formal --max-works 3`
2. `python scripts/build_register_clusters.py`
3. `python scripts/synthesize_briefs.py --cluster victorian-formal`
4. `python scripts/build_dataset.py --register victorian-formal`
5. `python scripts/train.py --register victorian-formal --max-steps 50 --eval-steps 25 --save-steps 25 --out runs/victorian-formal-smoke`
6. `python scripts/generate.py --register victorian-formal --brief "An afternoon visit to the parsonage."`
7. Run `scripts/style_profile.py` on outputs vs. base; pass criterion is ≥3 of 5 register axes shifted toward the cluster's target band, with no regression on a base instruction-following probe.

## Hardware

- Minimum: 12 GB VRAM
- Recommended: 16 GB VRAM (RTX 4080 / RTX 5070 Ti); 32 GB RAM; 100 GB SSD

## License

MIT — see `LICENSE`.
