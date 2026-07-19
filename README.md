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




# updates from Riccardo


## Task 2 – Preprocessing

Task 2 preprocesses the aligned 10% English–Italian sample produced in Task 1. Alignment is preserved by removing the complete sentence pair whenever either side is empty or starts with an XML-like tag.

The applied operations are:

- lowercase conversion;
- leading/trailing whitespace removal;
- repeated-whitespace normalisation;
- removal of empty aligned pairs;
- removal of aligned pairs containing XML-like lines.

Punctuation, numbers and stopwords are retained, and no stemming or lemmatisation is applied because these elements may contain information required for translation.

The main outputs are:

- `data/processed/task2_clean.en`;
- `data/processed/task2_clean.it`;
- `data/processed/task2_clean_parallel.csv`;
- preprocessing summaries and length statistics in `results/task2_outputs/`.

Final checks verify that the saved English and Italian files remain aligned.

## Task 3 – Neural Machine Translation

The Task 2 output is filtered to sentence pairs containing at most 20 words in both languages. The remaining data is split reproducibly into 70% training, 10% validation and 20% test sets using seed 42. The test set has not yet been used.

The word-level pipeline uses:

- whitespace tokenisation;
- separate 10,000-word English and Italian vocabularies built only on training data;
- `<pad>`, `<unk>`, `<sos>` and `<eos>` tokens;
- dynamic batch padding and packed source sequences;
- full teacher forcing;
- cross-entropy loss ignoring padding;
- Adam optimisation and gradient clipping;
- greedy decoding;
- BLEU and METEOR, reported on a 0–100 scale.

### Model development

The initial vanilla RNN encoder–decoder frequently collapsed to a single generic translation such as `la <unk>`. Increasing the number of epochs did not solve the problem and eventually caused overfitting.

The following changes were tested:

1. Vocabulary size increased from 5,000 to 10,000.
2. Invalid output tokens (`<pad>`, `<unk>`, `<sos>`) were masked during decoding.
3. The fixed encoder context was supplied at every decoder step.
4. Vanilla RNN units were replaced with LSTM units.
5. The LSTM hidden dimension was tuned using the validation set.
6. Random embeddings were compared with Word2Vec initialisation.

The main validation results are:

| Model | Validation loss | BLEU | METEOR |
|---|---:|---:|---:|
| Context-fed RNN | 4.2083 | 1.84 | 11.11 |
| LSTM, hidden size 128 | 3.8915 | 3.25 | 16.53 |
| LSTM, hidden size 256 | 3.7816 | 3.92 | 18.23 |
| LSTM + Word2Vec | **3.5596** | **4.17** | **19.23** |

The selected model is a single-layer context-fed LSTM with:

- embedding dimension: 64;
- hidden dimension: 256;
- learning rate: 0.001;
- batch size: 128;
- Word2Vec-initialised and fine-tuned embeddings;
- 20 maximum training epochs.

Separate English and Italian CBOW Word2Vec models were trained only on the training split using Gensim. Both covered all 9,996 non-special vocabulary tokens. The best Seq2Seq checkpoint was obtained at epoch 12; later epochs showed overfitting, but the training function restored the best validation checkpoint.

Selected checkpoint:

`models/task3/english_to_italian_context_lstm_large_word2vec.pt`

### Current limitations

The selected model produces more source-related words than the initial RNN, but translations still contain semantic errors, repetitions and generic Europarl expressions. The main limitation is the single fixed encoder context vector. Task 4 will address this using attention.

### Remaining Task 3 work

- final EN→IT test evaluation;
- Italian-to-English training and comparison;
- character-based model and word/character comparison;
- performance analysis by sentence length;
- qualitative error analysis.

I'm working on these while you go on with 4 and 5, we won't have conflicts

good luck with the next tasks!