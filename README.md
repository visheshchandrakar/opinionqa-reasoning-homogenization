# Does Reasoning Make It Worse?
### Test-Time Reasoning and Opinion Homogenization in Language Models — an extension of OpinionQA (Santurkar et al., 2023)

This repository contains the code, results and figures for a term paper that asks one question:

> **When a language model is allowed to "think" before answering an opinion question, do its answers become more diverse (closer to real people) or more collapsed onto a single answer?**

We answer it with a controlled experiment. Each model answers the same survey questions twice, with its built-in thinking mode **off** and **on**. The model weights, prompt and sampling settings are identical in both conditions. We then compare the model's answer distribution with the survey-weighted answers of real US respondents from the Pew Research Center's American Trends Panel.

**Main finding:** reasoning does **not** homogenize opinions. With thinking on, both models give *more* diverse answers, although both remain far more collapsed than real people. Representativeness is about the same (Qwen3-4B) or slightly better (SmolLM3-3B).

---

## Contents

- [Research design](#research-design)
- [Data](#data)
- [Models](#models)
- [Metrics](#metrics)
- [Results](#results)
- [Repository structure](#repository-structure)
- [How to run](#how-to-run)
- [Implementation notes and known issues](#implementation-notes-and-known-issues)
- [Limitations](#limitations)
- [Citation and acknowledgements](#citation-and-acknowledgements)

---

## Research design

| | |
|---|---|
| **Question pool** | OpinionQA, high-disagreement subset (498 questions, 14 Pew ATP waves, 2017–2021) |
| **Sample** | 100 questions, stratified across waves (seed 42) |
| **Conditions** | `no_think` (`enable_thinking=False`) vs. `think` (`enable_thinking=True`) |
| **Answers per question per condition** | K = 10 |
| **Sampling** | temperature 0.6, top-p 0.95, top-k 20 (the same in both conditions) |
| **Token budgets** | 512 tokens without thinking; 1,536 tokens with thinking (then *budget forcing*, see below) |
| **Comparison** | paired by question (think − no_think) |
| **Statistics** | mean difference with a 95% bootstrap CI (10,000 resamples of questions), Wilcoxon signed-rank test, Holm correction across the metrics of each model |

**Prompt.** We follow the OpinionQA default format. The question is followed by lettered answer options, including the survey's refusal option, and the instruction *"Answer the question by choosing one of the options above. Give your final answer in the format 'Answer: <letter>'."* The prompt is wrapped in each model's own chat template. The only difference between the two conditions is the `enable_thinking` flag.

**Budget forcing.** If a thinking trace reaches 1,536 tokens without closing `</think>`, we append *"I have to give my final answer now. </think> Answer:"* and let the model finish. These samples are flagged `truncated=True` and excluded in a robustness check. This affected 0.4% of Qwen3-4B's thinking samples and 1.6% of SmolLM3-3B's.

---

## Data

**OpinionQA** (Santurkar et al., 2023) turns Pew Research Center American Trends Panel (ATP) surveys into multiple-choice questions for language models, paired with the answers of thousands of real US respondents, their demographics and survey weights.

- Paper: <https://arxiv.org/abs/2303.17548>
- Code: <https://github.com/tatsu-lab/opinions_qa>
- Official data (CodaLab): <https://worksheets.codalab.org/worksheets/0x6fb693719477478aac73fc07db333f69>

**Data source used here.** The official CodaLab server could not be reached when this experiment was run, from Colab or from a local machine. We therefore used the processed OpinionQA distributions redistributed by the SubPOP project (Suh et al., ACL 2025, BSD-3 licence):

- <https://github.com/JosephJeesungSuh/subpop>, file `data/opinionqa/processed/opinionqa.csv`

This file holds the survey-weighted human answer distributions, overall and for each demographic group, for the OpinionQA **high-disagreement ("disagreement-500") subset**: 498 questions from 14 ATP waves (wave 27 is not included). Santurkar et al. use this subset for their steerability analysis. It suits a homogenization study well, because on these questions Americans disagree most, so a single dominant answer is clearly unrepresentative.

The notebook tries CodaLab first (`DATA_SOURCE = "auto"`) and falls back to the mirror automatically. When CodaLab works, setting `DATA_SOURCE = "codalab"` uses the full 1,498-question dataset and computes the weighted distributions directly from individual responses, following `helpers.extract_human_opinions` in the original repository.

> *Suggested wording for the Method section:* "We use the OpinionQA dataset (Santurkar et al., 2023), specifically its 500-question high-disagreement subset, with human opinion distributions as distributed via the SubPOP repository (Suh et al., 2025)."

---

## Models

| Model | Developer | Size | Role | Hugging Face |
|---|---|---|---|---|
| Qwen3-4B | Alibaba (Qwen team) | 4B | Primary | [`Qwen/Qwen3-4B`](https://huggingface.co/Qwen/Qwen3-4B) |
| SmolLM3-3B | Hugging Face | 3B | Replication in a second model family | [`HuggingFaceTB/SmolLM3-3B`](https://huggingface.co/HuggingFaceTB/SmolLM3-3B) |

Both are *hybrid reasoning* models: the same weights can answer with or without an explicit thinking trace, switched through the chat template's `enable_thinking` argument. This allows a clean within-model comparison, unlike comparing separate "chat" and "reasoning" models (e.g. DeepSeek-V3 vs. R1), which differ in more than reasoning. The two models come from different developers and training pipelines, so a shared effect is unlikely to be specific to one model family.

---

## Metrics

For each question, the K = 10 parsed answers form the model's opinion distribution **D_M** over the ordinal answer options. It is compared with the weighted human distribution **D_H**. Refusals are counted separately, as in OpinionQA.

| Metric | Definition | Reading |
|---|---|---|
| **Representativeness** | R = 1 − WD(D_M, D_H) / (n − 1): Wasserstein distance on the ordinal answer scale, normalised by the largest possible distance (the OpinionQA measure) | higher = closer to real people |
| **Normalised entropy** | H(D_M) / log n | 0 = all answers identical, 1 = evenly spread |
| **Modal share** | share of samples giving the most common answer | higher = more collapsed |
| **Distinct answers** | number of different options used | higher = more diverse |
| **Refusal rate** | share of samples choosing the refusal option | |
| **Parse-failure rate** | share of samples with no readable answer | quality check |

**Human K-sample baseline.** With only 10 samples, even a perfectly calibrated model looks somewhat collapsed. We therefore simulate a "model" that draws 10 answers from the true human distribution (500 Monte-Carlo repetitions per question). This shows how much of the observed narrowing is real homogenization and how much is small-sample noise.

---

## Results

100 questions × 10 answers × 2 conditions × 2 models = 4,000 answers. Parse failures: **0%** in every cell.

### Main results (think − no_think)

| Metric | Qwen3-4B: off → on | Δ [95% CI] | p (Holm) | SmolLM3-3B: off → on | Δ [95% CI] | p (Holm) |
|---|---|---|---|---|---|---|
| Representativeness | 0.724 → 0.717 | −0.007 [−0.032, +0.018] | 0.98 | 0.736 → 0.759 | +0.023 [+0.001, +0.046] | 0.041 |
| Normalised entropy | 0.121 → 0.194 | +0.074 [+0.012, +0.136] | 0.060 | 0.211 → 0.344 | +0.133 [+0.082, +0.186] | < 0.001 |
| Modal share | 0.928 → 0.874 | −0.054 [−0.094, −0.013] | 0.050 | 0.867 → 0.799 | −0.069 [−0.105, −0.032] | 0.001 |
| Distinct answers | 1.33 → 1.49 | +0.163 [0.000, +0.327] | 0.16 | 1.53 → 1.96 | +0.434 [+0.303, +0.576] | < 0.001 |
| Refusal rate | 0.029 → 0.103 | +0.074 [+0.029, +0.123] | 0.009 | 0.001 → 0.022 | +0.021 [+0.005, +0.046] | 0.14 |

### Pooled across both models

| Metric | Δ [95% CI] | p (Holm) |
|---|---|---|
| Representativeness | +0.008 [−0.010, +0.026] | 0.52 |
| Normalised entropy | +0.104 [+0.060, +0.148] | < 0.001 |
| Modal share | −0.062 [−0.092, −0.033] | < 0.001 |
| Distinct answers | +0.298 [+0.187, +0.414] | < 0.001 |
| Refusal rate | +0.048 [+0.023, +0.074] | 0.001 |

### Reference points

| | Representativeness | Normalised entropy | Modal share |
|---|---|---|---|
| Humans, 10 random draws (baseline) | 0.902 | 0.738 | 0.544 |
| Humans, full distribution | — | ≈ 0.85 | ≈ 0.48 |

### Robustness

- **Excluding truncated thinking traces:** same direction and similar size for every metric (Qwen3-4B entropy Δ = +0.073; SmolLM3-3B Δ = +0.133, p < 0.001).
- **Only questions with at least 5 valid answers in both conditions:** same pattern (Qwen3-4B entropy Δ = +0.079, p = 0.10; SmolLM3-3B Δ = +0.133, p < 0.001; SmolLM3-3B representativeness Δ = +0.023, p = 0.04).
- **Length of reasoning:** questions with longer thinking traces show a larger rise in entropy (Spearman r = 0.33, p = 0.001 for Qwen3-4B; r = 0.19, p = 0.06 for SmolLM3-3B).

### Interpretation

1. **Reasoning increases diversity, but only a little.** Both models are strongly collapsed without thinking: 87–93% of samples give the modal answer, against 54% for a human 10-draw baseline. Thinking reduces this collapse, but both models stay far from human diversity.
2. **More diverse is not the same as more representative.** The extra spread does not systematically move answers towards the human distribution. Representativeness is unchanged for Qwen3-4B and only slightly higher for SmolLM3-3B.
3. **Reasoning raises refusals for Qwen3-4B** (3% → 10%). The thinking trace often concludes that "as an AI, I have no personal view".
4. **Whose opinions?** In both conditions, both models are most representative of liberal, Democrat and highly educated respondents, and least representative of Republicans and very conservative respondents. This matches the original OpinionQA findings for other models.

---

## Repository structure

```
opinionqa-reasoning-homogenization/
├── README.md
├── opinionqa_reasoning_analysis.ipynb   # the full pipeline (Colab-ready)
└── results/                             # outputs of the q100_k10 run
    ├── questions.jsonl                  # the 100 sampled questions
    ├── summary.txt                      # plain-language summary of the main table
    ├── generations/                     # raw model outputs, one JSON line per sample
    │   ├── Qwen3-4B__no_think.jsonl
    │   ├── Qwen3-4B__think.jsonl
    │   ├── SmolLM3-3B__no_think.jsonl
    │   └── SmolLM3-3B__think.jsonl
    ├── tables/
    │   ├── main_results.csv / main_results.tex
    │   ├── pooled_results.csv
    │   ├── robustness.csv
    │   ├── length_vs_entropy.csv
    │   ├── per_question_metrics.csv
    │   ├── group_representativeness.csv
    │   └── most_represented_group.csv
    └── figures/                         # PNG and PDF
        ├── fig1_main_metrics            # means with 95% CIs and human baselines
        ├── fig2_entropy_scatter         # per-question entropy, off vs. on
        ├── fig3_rep_diff                # distribution of Δ representativeness
        ├── fig4_groups_POLIDEOLOGY      # representativeness by political ideology
        └── fig4_groups_POLPARTY         # representativeness by party
```

Each line in `generations/*.jsonl` contains: `model`, `condition`, `qkey`, `sample`, `answer_idx` (parsed option index, `null` if unparsed), `n_opts`, `n_refs`, `n_gen_tokens`, `think_chars`, `truncated`, and `raw` (the last 4,000 characters of the output).

---

## How to run

The notebook is written for **Google Colab with a free T4 GPU**.

1. Open `opinionqa_reasoning_analysis.ipynb` in Colab (*File → Upload notebook*).
2. *Runtime → Change runtime type → T4 GPU*.
3. In the configuration cell, set `SMOKE_TEST = True` and choose *Runtime → Run all*. This is a 3-question test and takes about 10 minutes, most of it installing packages and downloading models.
4. Set `SMOKE_TEST = False` and run all again for the full experiment.

**Main settings** (configuration cell):

| Setting | Default | Meaning |
|---|---|---|
| `MODELS` | Qwen3-4B, SmolLM3-3B | models to compare |
| `N_QUESTIONS` / `N_SAMPLES` | 100 / 10 | questions and answers per condition |
| `TEMPERATURE` / `TOP_P` / `TOP_K` | 0.6 / 0.95 / 20 | sampling (same in both conditions) |
| `MAX_TOKENS_NO_THINK` / `MAX_TOKENS_THINK` | 512 / 1536 | output budgets |
| `ENGINE` | `"auto"` | vLLM if it works, else Hugging Face transformers |
| `DATA_SOURCE` | `"auto"` | CodaLab if reachable, else the SubPOP mirror |
| `USE_DRIVE` | `True` | save outputs to Google Drive (recommended) |

**Run time on a T4 with vLLM:** about 1 minute per model without thinking, 30–45 minutes per model with thinking, plus about 10 minutes per model for downloading and loading. The full run takes about 1.5–2 hours.

**Resuming.** Generations are written to disk every 5–20 questions. If Colab disconnects, run all cells again: finished questions are skipped. Keep `USE_DRIVE = True` so outputs survive a lost runtime.

---

## Implementation notes and known issues

- **vLLM in Colab.** Installing vLLM upgrades PyTorch, which leaves Colab's preinstalled `torchaudio` mismatched and breaks `import transformers`. The notebook uninstalls `torchaudio` (neither model needs it). vLLM's engine also fails inside notebooks because it calls `sys.stdout.fileno()`. The notebook runs the engine in-process (`VLLM_ENABLE_V1_MULTIPROCESSING=0`) and disables that redirection.
- **GPU memory between models.** vLLM does not release GPU memory when the first model finishes. The second model then fails with *CUDA out of memory*. Fix: *Runtime → Restart session and run all*. The first model's outputs are kept and skipped.
- **T4 support.** The T4 (compute capability 7.5) cannot use FlashAttention-2. vLLM falls back to Triton attention automatically and prints a harmless error line about FA2.
- **Output budget without thinking.** A limit of 64 tokens was too short: Qwen3-4B often writes a short explanation before the answer letter, which left 24% of answers unreadable. With 512 tokens the parse-failure rate is 0%. Both models' no-thinking answers were regenerated with this setting.
- **Unique question IDs.** A few Pew keys contain several sub-questions (e.g. `IDIMPORT_W43`). These are given unique IDs (`IDIMPORT_W43#2`, …).

---

## Limitations

- **Small models only.** The results are for 3–4B models on a free GPU; larger models may behave differently.
- **Sampling-based distributions.** Opinion distributions are estimated from 10 samples per question, which is coarse. The human 10-draw baseline helps read the numbers, but more samples (e.g. 20–50) would give more precise estimates.
- **One question subset.** The high-disagreement subset of OpinionQA was used, not all 1,498 questions.
- **One prompt format.** Only the default OpinionQA prompt was used; no steering or persona prompts.
- **Refusals.** Refusals are excluded from D_M, as in OpinionQA, so the higher refusal rate with thinking on slightly reduces the number of valid answers for Qwen3-4B.

---

## Citation and acknowledgements

If you use this work, please also cite the resources it builds on:

```bibtex
@inproceedings{santurkar2023whose,
  title     = {Whose Opinions Do Language Models Reflect?},
  author    = {Santurkar, Shibani and Durmus, Esin and Ladhak, Faisal and Lee, Cinoo and Liang, Percy and Hashimoto, Tatsunori},
  booktitle = {Proceedings of the 40th International Conference on Machine Learning (ICML)},
  year      = {2023}
}

@inproceedings{suh2025subpop,
  title     = {Language Model Fine-Tuning on Scaled Survey Data for Predicting Distributions of Public Opinions},
  author    = {Suh, Joseph and Jahanparast, Erfan and Moon, Suhong and Kang, Minwoo and Chang, Serina},
  booktitle = {Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (ACL)},
  year      = {2025}
}
```

The survey data originates from the **Pew Research Center American Trends Panel**. Pew Research Center bears no responsibility for the analyses or interpretations presented here. Model weights are from the Qwen team (Alibaba) and Hugging Face.
