# English–Italian Machine Translation

NLP Project 2.2 — Vasilis & Riccardo.
RNN sequence-to-sequence translation on the Europarl v7 `it-en` parallel corpus.

## Note on the data (please read first)

I've set this repo up so you don't have to download anything. Here's the situation: the
full Europarl corpus is two files of 285 MB and 311 MB, and GitHub hard-rejects any file
over 100 MB, so the raw corpus simply cannot live in the repo — that part isn't a choice,
it's a platform limit. What I did instead is commit the **10% sample** that the project
brief tells us to train on (190,912 aligned sentence pairs), gzipped down to about 20 MB,
which fits comfortably. This costs us nothing, because the raw corpus is only ever read by
Task 1 to compute the corpus statistics and cut that sample — and both of those outputs
are already committed here (`results/task1_outputs/` and `data/sample/`). Tasks 2 through 5
read *only* the sample. So: **clone the repo, run the one `gunzip` command below, and you
have everything you need to work.** The sample is cut with a fixed seed (42) and its row
indices are saved in `data/sample/sample_idx.npy`, so we're both guaranteed to be training
on the exact same data — which also covers the reproducibility the brief grades us on. The
only reason you'd ever need the raw corpus is if you want to re-run Task 1 yourself, and
there's a `wget` one-liner for that further down. One last thing: please **don't commit the
uncompressed `sample.en` / `sample.it` or an `it-en/` folder** — the `.gitignore` already
blocks them, and that's deliberate, so let's keep it that way.

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

## Working together

Jupyter notebooks merge badly in git — if we both edit the same `.ipynb`, git can't
reconcile them and we get a wall of unreadable JSON conflicts. Easiest fix is to agree who
owns which notebook and stay out of each other's files. Usual rhythm:

```bash
git pull                      # always, before starting work
# ... work ...
git add -A && git commit -m "what I did" && git push
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
