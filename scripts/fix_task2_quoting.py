"""Repair the Task 2 plain-text outputs.

Task 2 saved the cleaned sentences with pandas `.to_csv()`, which applies CSV
quoting rules: any field containing a comma is wrapped in double quotes, and any
real double quote inside a field is escaped by doubling it. The result is that
64% of the lines in task2_clean.{en,it} carry literal quote characters that were
never part of the text, e.g.

    "madam president, on a point of order."      <- should have no outer quotes
    my vote was ""in favour"" .                  <- should be "in favour"

Because Task 3 reads those files as plain text, the models were trained on the
corrupted strings, wasting ~24,000 vocabulary entries on phantom duplicates.

The sibling file task2_clean_parallel.csv is a *correct* CSV, so the clean text
is fully recoverable: we simply parse it and rewrite the plain-text files
without any CSV encoding. There is no need to re-run Task 2.

Usage:  python scripts/fix_task2_quoting.py
"""

from pathlib import Path

import pandas as pd


def find_project_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "data" / "processed").exists():
            return candidate
    raise FileNotFoundError("Could not locate the project root.")


def main() -> None:
    root = find_project_root(Path(__file__).resolve().parent)
    processed = root / "data" / "processed"

    csv_path = processed / "task2_clean_parallel.csv"
    en_path = processed / "task2_clean.en"
    it_path = processed / "task2_clean.it"

    frame = pd.read_csv(csv_path, encoding="utf-8", keep_default_na=False)
    if list(frame.columns) != ["en", "it"]:
        raise ValueError(f"Unexpected columns in {csv_path.name}: {list(frame.columns)}")

    before = sum(
        1
        for line in en_path.open(encoding="utf-8")
        if line.startswith('"') or '""' in line
    )

    # Write plain text: one sentence per line, no CSV encoding of any kind.
    for path, column in ((en_path, "en"), (it_path, "it")):
        text = "\n".join(frame[column].astype(str)) + "\n"
        path.write_text(text, encoding="utf-8")

    after = sum(
        1
        for line in en_path.open(encoding="utf-8")
        if line.startswith('"') or '""' in line
    )

    english = [line.rstrip("\n") for line in en_path.open(encoding="utf-8")]
    italian = [line.rstrip("\n") for line in it_path.open(encoding="utf-8")]
    assert len(english) == len(italian) == len(frame), "Alignment broken while rewriting."
    assert all(s.strip() for s in english), "Empty English line produced."
    assert all(s.strip() for s in italian), "Empty Italian line produced."

    print(f"pairs rewritten     : {len(frame):,}")
    print(f"quote-corrupted lines: {before:,} -> {after:,}")
    print(f"example              : {english[0]}")


if __name__ == "__main__":
    main()
