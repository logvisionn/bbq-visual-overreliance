# Visual Over-reliance in Large Vision-Language Models

Full-scale evaluation of **visual over-reliance** in Qwen2-VL on the **BBQ bias benchmark** (58,492 items), with a three-condition design that decomposes the total visual effect into an **architecture-mode penalty** and an **image-content effect**.

> **Bachelor practical work · JKU Linz · Supervisor: Deepak Kumar, Institute of Computational Perception**
> Mykola Len — Matriculation No. 12340329

📄 **[Read the write-up (PDF)](./Mykola%20Len%20Visual%20Overreliance%20writeup.pdf)**

---

## TL;DR

- We pair each of the 58,492 BBQ items [Parrish et al. 2022] with (a) no image, (b) a neutral blank gray image, and (c) an SDXL-turbo-generated image of the scenario, and run all three through Qwen2-VL-2B and Qwen2-VL-7B [Wang et al. 2024] — 350,952 inference calls total.
- **Key finding:** on Qwen2-VL-7B, merely activating the vision encoder with a blank gray image costs **−10.6 pp** of accuracy. Adding real image content on top costs a further **−4.2 pp**. The "blank image penalty" is ~72% of the total drop.
- What prior work called *visual over-reliance* therefore conflates two phenomena. Only the residual **C2 → C3** content component tracks the BBQ bias score `sAMB`, concentrates in ambiguous contexts, and scales with visually inferable categories (physical appearance, age, disability status, gender identity).

## Experimental design

| Condition | Text | Image | What it isolates |
|-----------|------|-------|------------------|
| **C1** | BBQ item | *(none)* | Text-only baseline |
| **C2** | BBQ item | Uniform RGB(128,128,128) 512×512 | C1→C2 = architecture-mode penalty |
| **C3** | BBQ item | SDXL-turbo rendering of the scenario | C2→C3 = image-content effect |

Six runs: `{Qwen2-VL-2B, Qwen2-VL-7B} × {C1, C2, C3}`. All 58,492 SDXL images were generated once in a single 10.6 h pass and reused across the two C3 inference runs, so C3 differences between model runs reflect the model, not the stimulus.

## Headline numbers

| Model | C1 acc. | C2 acc. | C3 acc. | Total drop | Arch. penalty (C1→C2) | Content effect (C2→C3) |
|-------|--------:|--------:|--------:|-----------:|----------------------:|------------------------:|
| Qwen2-VL-2B | 0.5463 | 0.5239 | 0.4981 | −4.8 pp | −2.2 pp (46%) | −2.6 pp (54%) |
| Qwen2-VL-7B | 0.8044 | 0.6981 | 0.6563 | −14.8 pp | **−10.6 pp (72%)** | −4.2 pp (28%) |

Paired bootstrap (10,000 resamples) rejects H₀ for every C2-vs-C3 comparison in ambiguous contexts at p < 0.0001. The only test that does **not** reject H₀ is the 7B C1-vs-C3 *disambiguating* split (Δ = +0.002, p = 0.083) — strong evidence that the content-driven effect is ambiguity-triggered.

## Repository layout

```
.
├── Mykola Len Visual Overreliance writeup.pdf   # write-up — read this first
│
├── run_full_experiment.py                  # all six {model × condition} inference jobs
├── analyse_results.py                      # accuracy, sAMB/sDIS, bootstrap, per-category
│
├── results/
│   ├── bbq_items.csv                       # 58,492 canonical BBQ items (shuffle seed 42)
│   ├── run_config.json                     # run metadata
│   ├── preds_qwen2b_c1.csv   …   preds_qwen7b_c3.csv
│   └── analysis/
│       ├── accuracy_summary.csv
│       ├── bias_scores.csv                 # sAMB / sDIS per (model, condition, category)
│       ├── bootstrap_tests.csv
│       ├── c1_vs_c2_comparison.csv
│       ├── per_category_stats.csv
│       └── summary_report.txt              # data coverage + key statistics
│
├── LICENSE                                 # MIT
├── .gitignore
└── README.md
```

Note: the 2.9 GB of SDXL-generated images are **not** committed to the repository. They are deterministic outputs of the `stabilityai/sdxl-turbo` model applied to the prompts in `results/bbq_items.csv`, and can be regenerated from scratch using step 2 below.

## Reproducing the results

### 1. Environment

Tested on Ubuntu 22.04 + NVIDIA RTX 4090 (24 GB VRAM) + Python 3.10:

```bash
pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121
pip install transformers accelerate bitsandbytes
pip install diffusers safetensors
pip install pandas numpy scipy matplotlib pillow
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
- `results/analysis/bias_scores.csv` — sAMB and sDIS per (model, condition, category)
- `results/analysis/bootstrap_tests.csv` — paired bootstrap tests on accuracy differences
- `results/analysis/c1_vs_c2_comparison.csv` — architecture-mode isolation
- `results/analysis/per_category_stats.csv` — per-category means and stds
- `results/analysis/summary_report.txt` — human-readable roll-up

## Methodology at a glance

- **BBQ bias scores** follow Parrish et al. [2022]: `sDIS = 2·(n_biased / n_non_unknown) − 1` on disambiguating items; `sAMB = (1 − Accuracy) · sDIS_on_ambiguous` on ambiguous items.
- **Null hypothesis testing** uses a **paired bootstrap** (10,000 resamples, 95% percentile CI) on accuracy differences, replacing the McNemar test used in the pilot.
- **Per-category variance** is reported as mean ± std across the 11 BBQ bias categories for every (model, condition) cell.
- **Paired intersection** handles the 481 IMAGE_MISSING rows cleanly: any comparison that involves C3 restricts to the 58,011 rows where a valid image was generated for *both* sides of the comparison.

## Limitations

1. 4-bit NF4 quantisation may amplify noise in small effect sizes; full-precision inference wasn't feasible on a 24 GB GPU for the 7B model.
2. Only two model sizes from one LVLM family (Qwen2-VL). The decomposition ratio (architecture-mode vs. content effect) has not been verified on LLaVA, PaliGemma, Gemma 3, or InternVL.
3. The "architecture-mode penalty" is an *effect*, not a *mechanism*. The paper speculates about M-RoPE and attention dilution; this has not yet been causally tested.

## Citation

```bibtex
@misc{len2026bbqvisualoverreliance,
  author       = {Mykola Len},
  title        = {Visual Over-reliance in Large Vision-Language Models:
                  A Decomposed Analysis on the BBQ Bias Benchmark},
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
