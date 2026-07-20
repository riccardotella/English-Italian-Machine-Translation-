"""Build an Italian-English-Swedish parallel corpus for the Task 5 pivot.

Europarl ships bilingual corpora (it-en, sv-en) that are aligned independently:
line n of it-en is not the same sentence as line n of sv-en. To evaluate
Italian->Swedish translation we need genuine IT-SV reference pairs, so we build
them by joining the two corpora on their shared English side.

Both corpora are transcripts of the same European Parliament proceedings, so a
large fraction of English sentences appear verbatim in both. Normalising
whitespace and case (the same preprocessing Task 2 applies) and matching on the
English string recovers ~157k IT-EN-SV triples.

The output is gzipped and committed so that Task 5 can run from a fresh clone
without downloading the 560 MB Swedish corpus.

Usage:  python scripts/build_pivot_triples.py
"""

import gzip
import re
from pathlib import Path

MAX_TOKENS = 20  # matches the cap used in Tasks 3-4


def normalise(text: str) -> str:
    """Task 2's preprocessing: collapse whitespace, strip, lowercase."""
    return re.sub(r"\s+", " ", text).strip().lower()


def find_project_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "data" / "processed").exists():
            return candidate
    raise FileNotFoundError("Could not locate the project root.")


def main() -> None:
    root = find_project_root(Path(__file__).resolve().parent)
    sv_dir = root / "sv-en"
    out_dir = root / "data" / "pivot"
    out_dir.mkdir(parents=True, exist_ok=True)

    sv_en = sv_dir / "europarl-v7.sv-en.en"
    sv_sv = sv_dir / "europarl-v7.sv-en.sv"
    if not (sv_en.exists() and sv_sv.exists()):
        raise FileNotFoundError(
            f"Swedish corpus not found in {sv_dir}. Download it with:\n"
            "  wget https://www.statmt.org/europarl/v7/sv-en.tgz\n"
            "  mkdir -p sv-en && tar -xzf sv-en.tgz -C sv-en"
        )

    # English -> Swedish lookup. Keep the first occurrence of each English
    # sentence; duplicates in Europarl are near-identical boilerplate.
    english_to_swedish: dict[str, str] = {}
    with sv_en.open(encoding="utf-8") as fe, sv_sv.open(encoding="utf-8") as fs:
        for english_line, swedish_line in zip(fe, fs):
            english, swedish = normalise(english_line), normalise(swedish_line)
            if english and swedish:
                english_to_swedish.setdefault(english, swedish)
    print(f"unique English sentences in sv-en : {len(english_to_swedish):,}")

    processed = root / "data" / "processed"
    english_side = [l.rstrip("\n") for l in (processed / "task2_clean.en").open(encoding="utf-8")]
    italian_side = [l.rstrip("\n") for l in (processed / "task2_clean.it").open(encoding="utf-8")]
    assert len(english_side) == len(italian_side), "Task 2 files are not aligned."
    print(f"IT-EN pairs from Task 2           : {len(english_side):,}")

    triples = [
        (italian, english, english_to_swedish[english])
        for english, italian in zip(english_side, italian_side)
        if english in english_to_swedish
    ]
    print(f"IT-EN-SV triples                  : {len(triples):,} "
          f"({100 * len(triples) / len(english_side):.1f}%)")

    triples = [
        t for t in triples
        if all(len(side.split()) <= MAX_TOKENS for side in t)
    ]
    print(f"  after <= {MAX_TOKENS}-token filter        : {len(triples):,}")

    for index, language in enumerate(("it", "en", "sv")):
        path = out_dir / f"triples.{language}.gz"
        with gzip.open(path, "wt", encoding="utf-8") as f:
            f.write("\n".join(t[index] for t in triples) + "\n")
        print(f"  wrote {path.relative_to(root)}  "
              f"({path.stat().st_size / 1_048_576:.1f} MB)")


if __name__ == "__main__":
    main()
