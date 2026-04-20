# Visual Over-reliance in Large Vision-Language Models

**From Architecture-Mode Penalty to Premature Commitment: Two Decompositions on the BBQ Benchmark**

Full-scale evaluation of **visual over-reliance** in Qwen2-VL on the **BBQ bias benchmark** (58,492 items). Using a three-condition experimental design, we decompose the total visual effect along two orthogonal axes: (i) an **architecture-mode penalty** (C1→C2) vs. an **image-content effect** (C2→C3), and (ii) within the ambiguous-context bias score sAMB, a **commitment-rate component** vs. a **stereotype-ratio component**.

> **Bachelor practical work · JKU Linz · Supervisor: Deepak Kumar, Institute of Computational Perception**
> Mykola Len — Matriculation No. 12340329

📄 **[Read the write-up (PDF)](./Mykola%20Len%20Visual%20Overreliance%20writeup.pdf)**

---

## Contributions

1. **Architecture-mode penalty** (first decomposition). On Qwen2-VL-7B, merely activating the vision encoder with a blank gray image costs **−10.6 pp** accuracy. Adding real image content on top costs a further **−4.2 pp**. The "blank image penalty" accounts for ~72% of the total visual drop.
2. **The sAMB rise is not a stereotype rise** (second decomposition). When sAMB is factored back into its (1 − Accuracy) multiplier and its raw stereotype-commitment ratio sDIS<sub>amb</sub>, sDIS<sub>amb</sub> *falls or stays flat* in 9 of 11 bias categories on 7B (7B pooled: +0.295 → +0.179 C1→C3). The apparent sAMB rise is entirely driven by the accuracy component.
3. **Interpretation: premature commitment under visual input.** Images do not make the model pick more stereotyped answers; they make the model pick a named answer at all, when the correct response is to withhold judgment. On Qwen2-VL-7B, commitments on ambiguous items nearly double (9,722 → 18,310). Image-conditioned abstention is therefore a more promising intervention target than stereotype-specific answer reweighting.

## Experimental design

| Condition | Text | Image | What it isolates |
|-----------|------|-------|------------------|
| **C1** | BBQ item | *(none)* | Text-only baseline |
| **C2** | BBQ item | Uniform RGB(128,128,128) 512×512 | C1→C2 = architecture-mode penalty |
| **C3** | BBQ item | SDXL-turbo rendering of the scenario | C2→C3 = image-content effect |

Six runs: `{Qwen2-VL-2B, Qwen2-VL-7B} × {C1, C2, C3}`. All 58,492 SDXL images were generated once in a single 10.6 h pass and reused across the two C3 inference runs, so C3 differences between model runs reflect the model, not the stimulus.

## Headline numbers

### Decomposition 1: architecture-mode vs. image-content

| Model | C1 acc. | C2 acc. | C3 acc. | Total drop | Arch. penalty (C1→C2) | Content effect (C2→C3) |
|-------|--------:|--------:|--------:|-----------:|----------------------:|------------------------:|
| Qwen2-VL-2B | 0.5463 | 0.5239 | 0.4981 | −4.8 pp | −2.2 pp (46%) | −2.6 pp (54%) |
| Qwen2-VL-7B | 0.8044 | 0.6981 | 0.6563 | −14.8 pp | **−10.6 pp (72%)** | −4.2 pp (28%) |

Paired bootstrap (10,000 resamples) rejects H₀ for every ambiguous-context comparison at p < 0.0001. The only test that does **not** reject H₀ is the 7B C1-vs-C3 *disambiguating* split (Δ = +0.002, p = 0.083) — strong evidence that the image-content effect is ambiguity-triggered.

### Decomposition 2: sAMB = (1 − Accuracy<sub>amb</sub>) · sDIS<sub>amb</sub>

| Model | Cond. | Acc. (amb) | 1 − Acc | sDIS<sub>amb</sub> | sAMB | n_commit |
|-------|-------|-----------:|--------:|-------------------:|-----:|---------:|
| Qwen2-VL-2B | C1 | 0.310 | 0.690 | +0.115 | +0.079 | 20,187 |
|             | C2 | 0.255 | 0.745 | +0.102 | +0.076 | 21,781 |
|             | C3 | 0.169 | 0.831 | +0.101 | +0.084 | 24,101 |
| Qwen2-VL-7B | C1 | 0.668 | 0.332 | +0.295 | +0.098 |  9,722 |
|             | C2 | 0.443 | 0.557 | +0.197 | +0.110 | 16,298 |
|             | C3 | 0.369 | 0.631 | +0.179 | +0.113 | 18,310 |

On Qwen2-VL-7B, **sDIS<sub>amb</sub> falls from +0.295 to +0.179 C1→C3**; the sAMB rise from +0.098 to +0.113 is entirely driven by the (1 − Accuracy) multiplier. Per-category sDIS<sub>amb</sub> drifts C1→C3 are negative in 9 of 11 BBQ categories (full table in the write-up and in `results/analysis/bias_scores.csv`).

## Repository layout

```
.
├── Mykola Len Visual Overreliance writeup.pdf  # write-up — read this first
│
├── run_full_experiment.py                      # all six {model × condition} inference jobs
├── analyse_results.py                          # accuracy, sAMB/sDIS, bootstrap, per-category
├── requirements.txt                            # pinned dependency versions
│
├── results/
│   ├── bbq_items.csv                           # 58,492 canonical BBQ items (shuffle seed 42)
│   ├── run_config.json                         # run metadata
│   ├── preds_qwen2b_c1.csv   …   preds_qwen7b_c3.csv
│   └── analysis/
│       ├── accuracy_summary.csv
│       ├── bias_scores.csv                     # sAMB / sDIS / sDIS_amb components per category
│       ├── bootstrap_tests.csv
│       ├── c1_vs_c2_comparison.csv
│       ├── per_category_stats.csv
│       └── summary_report.txt                  # data coverage + key statistics
│
├── LICENSE                                     # MIT
├── .gitignore
└── README.md
```

Note: the 2.9 GB of SDXL-generated images are **not** committed to the repository. They are deterministic outputs of the `stabilityai/sdxl-turbo` model applied to the prompts in `results/bbq_items.csv`, and can be regenerated from scratch using step 2 below.

## Reproducing the results

### 1. Environment

Tested on Ubuntu 22.04 + NVIDIA RTX 4090 (24 GB VRAM) + Python 3.10:

```bash
# PyTorch first (CUDA 12.1 build):
pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121

# Then the rest:
pip install -r requirements.txt
```

The BBQ dataset loads from Hugging Face (`heegyu/bbq`) on first run; no manual download required.

### 2. Run the full pipeline

The script orchestrates everything: it generates the 58,492 SDXL images once (~10.6 h on a 4090), then runs all six `{model × condition}` inference jobs (~4–6 h each):

```bash
python run_full_experiment.py
```

Key flags:

| Flag | Effect |
|------|--------|
| `--skip-images` | Skip image generation (use if images already exist on disk) |
| `--only-model {2b,7b}` | Run only one model |
| `--only-cond {c1,c2,c3}` | Run only one condition |

Useful partial runs:

```bash
# Only the 7B model, all three conditions (requires images already generated)
python run_full_experiment.py --skip-images --only-model 7b

# Only C1 (text-only) across both models — no images needed at all
python run_full_experiment.py --skip-images --only-cond c1
```

Outputs land in `results/`:
- `results/images_exp/` — 58,492 SDXL JPEGs (gitignored)
- `results/preds_{model}_{cond}.csv` — one file per run, with per-item prediction and correctness

### 3. Analyse results

```bash
python analyse_results.py
```

Reads all six `preds_*.csv` files, writes:
- `results/analysis/accuracy_summary.csv` — overall accuracy per (model, condition, context)
- `results/analysis/bias_scores.csv` — sAMB, sDIS, and component counts (`n_biased_amb`, `n_non_unk_amb`) per (model, condition, category), from which sDIS<sub>amb</sub> can be recovered as `2·(n_biased_amb / n_non_unk_amb) − 1`
- `results/analysis/bootstrap_tests.csv` — paired bootstrap tests on accuracy differences
- `results/analysis/c1_vs_c2_comparison.csv` — architecture-mode isolation
- `results/analysis/per_category_stats.csv` — per-category means and stds
- `results/analysis/summary_report.txt` — human-readable roll-up

### Minimal rerun (fastest way to validate the paper's numbers)

All analysis outputs are committed under `results/analysis/` and the six prediction CSVs are in `results/`. To re-derive the paper's headline numbers without running any inference, from the repository root:

```bash
pip install pandas numpy
python analyse_results.py   # rebuilds results/analysis/* from the committed preds_*.csv
```

This takes well under a minute and is sufficient to verify every table and bootstrap value in the write-up. Full reproduction (image generation + inference) takes roughly 35–40 GPU-hours on an RTX 4090.

## Methodology at a glance

- **BBQ bias scores** follow Parrish et al. [2022]: `sDIS = 2·(n_biased / n_non_unknown) − 1` on disambiguating items; `sAMB = (1 − Accuracy) · sDIS_on_ambiguous` on ambiguous items.
- **sAMB decomposition.** Because sAMB is a product of two quantities, a rise in sAMB can reflect either (i) the model becoming more stereotype-aligned in its commitments (sDIS<sub>amb</sub> rising) or (ii) the model committing more often at an unchanged bias ratio ((1 − Accuracy) rising). We report both components separately and find that mechanism (ii) dominates in our data.
- **Null hypothesis testing** uses a **paired bootstrap** (10,000 resamples, 95% percentile CI) on accuracy differences, replacing the McNemar test used in the pilot.
- **Per-category variance** is reported as mean ± std across the 11 BBQ bias categories for every (model, condition) cell.
- **Paired intersection** handles the 481 IMAGE_MISSING rows cleanly: any comparison that involves C3 restricts to the 58,011 rows where a valid image was generated for *both* sides of the comparison.

## Limitations

1. 4-bit NF4 quantisation may amplify noise in small effect sizes; full-precision inference wasn't feasible on a 24 GB GPU for the 7B model.
2. Only two model sizes from one LVLM family (Qwen2-VL). The architecture-mode vs. content-effect split, and the premature-commitment interpretation, have not been verified on LLaVA, PaliGemma, Gemma 3, or InternVL.
3. The "architecture-mode penalty" is an *effect*, not a *mechanism*. The paper speculates about M-RoPE and attention dilution; this has not yet been causally tested.
4. The sAMB decomposition depends on the specific multiplicative structure of the BBQ bias score as defined by Parrish et al. Alternative scoring conventions would produce a different decomposition and potentially a different narrative.

## Citation

```bibtex
@misc{len2026bbqvisualoverreliance,
  author       = {Mykola Len},
  title        = {Visual Over-reliance in Large Vision-Language Models:
                  From Architecture-Mode Penalty to Premature Commitment},
  year         = {2026},
  howpublished = {Bachelor practical work, Johannes Kepler University Linz},
  note         = {\url{https://github.com/logvisionn/bbq-visual-overreliance}}
}
```

## References

- Parrish, A. et al. (2022). *BBQ: A Hand-Built Bias Benchmark for Question Answering.* Findings of ACL 2022.
- Wang, P. et al. (2024). *Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution.* arXiv:2409.12191.
- Liu, H. et al. (2023). *Visual Instruction Tuning (LLaVA).* NeurIPS 2023 Oral.
- Podell, D. et al. (2023). *SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis.* ICLR 2024.
- Dettmers, T. et al. (2023). *QLoRA: Efficient Finetuning of Quantized LLMs.* NeurIPS 2023.
- Bianchi, F. et al. (2023). *Easily Accessible Text-to-Image Generation Amplifies Demographic Stereotypes at Large Scale.* ACM FAccT 2023.

Full reference list in the [write-up PDF](./Mykola%20Len%20Visual%20Overreliance%20writeup.pdf).

## License

MIT — see [LICENSE](./LICENSE). Datasets referenced (BBQ, Qwen2-VL weights, SDXL-turbo weights) are subject to their own licences.
