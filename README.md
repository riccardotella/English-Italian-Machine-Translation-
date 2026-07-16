# English–Italian Machine Translation

NLP Project 2.2 — Vasilis & Riccardo.
RNN sequence-to-sequence translation on the Europarl v7 `it-en` parallel corpus.

## Quick start (no downloads needed)

The **10% sample that all models train on is included in this repo** (gzipped).
After cloning, unpack it:

```bash
gunzip -k data/sample/sample.en.gz data/sample/sample.it.gz
```

That gives you `data/sample/sample.en` and `data/sample/sample.it` — 190,912 aligned
sentence pairs. Tasks 2–5 read only these files, so this is all you need.

## About the raw corpus

The full corpus (`it-en/europarl-v7.it-en.{en,it}`, ~595 MB) is **not** in the repo:
GitHub rejects files over 100 MB. You only need it if you want to re-run Task 1
from scratch. To get it:

```bash
wget https://www.statmt.org/europarl/v7/it-en.tgz
mkdir -p it-en && tar -xzf it-en.tgz -C it-en
```

Task 1's outputs are already committed (`results/task1_outputs/`), and the sample is
regenerated deterministically (`seed=42`, indices saved in `data/sample/sample_idx.npy`),
so re-running Task 1 reproduces the identical sample.

## Layout

```
notebooks/     one notebook per task
data/sample/   the 10% sample (gzipped) + sample indices
results/       figures and metrics per task, used in the report
it-en/         raw corpus (git-ignored, download separately)
```

## Corpus at a glance (full corpus, from Task 1)

| | English | Italian |
|---|---|---|
| Sentence pairs | 1,909,115 | 1,909,115 |
| Total tokens | 49.7 M | 48.0 M |
| Vocabulary (types) | 275,537 | **426,025** |
| Mean sentence length | 26.05 | 25.13 |

Italian's vocabulary is ~55% larger despite fewer tokens — morphological richness,
which is the main argument for the character-based model in Task 3.
