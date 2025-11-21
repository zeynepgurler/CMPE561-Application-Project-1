"""
Build a plain-text Turkish corpus using Wikipedia API,
using the titles from your existing large gazetteer.

For each title:
    - Fetch the page using Wikipedia API
    - Extract the main text (page.text)
    - Save it as a .txt file in CORPUS_DIR

Later, safe_vocab builder can read this folder.
"""

import wikipediaapi
import time
from pathlib import Path
import re
import os


# -------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------

GAZETTEER_FILE = "normalization_resources/tr_gazetteer_large.txt"      # existing gazetteer (106k items)
CORPUS_DIR = "data/wiki_text"                  # corpus output directory
SLEEP_BETWEEN_REQUESTS = 0.2                   # to avoid rate-limiting (seconds)


# -------------------------------------------------------------
# Wikipedia client
# -------------------------------------------------------------
wiki = wikipediaapi.Wikipedia(
    language="tr",
    user_agent="NLP-Normalization-Project/1.0 (contact: example@example.com)"
)


# -------------------------------------------------------------
# Helpers
# -------------------------------------------------------------
def safe_filename(title: str) -> str:
    """
    Turn a page title into a safe filename:
    Replace spaces, remove invalid characters.
    """
    title = title.strip()
    title = re.sub(r"[^\w\- ]+", "", title)
    title = title.replace(" ", "_")
    return title + ".txt"


def read_gazetteer(path: str) -> list:
    """
    Read gazetteer file (one title per line).
    """
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


# -------------------------------------------------------------
# Main corpus builder
# -------------------------------------------------------------
def build_corpus():
    corpus_dir = Path(CORPUS_DIR)
    corpus_dir.mkdir(parents=True, exist_ok=True)

    titles = read_gazetteer(GAZETTEER_FILE)
    print(f"[INFO] Loaded {len(titles)} gazetteer titles.")

    saved = 0
    skipped = 0

    for i, title in enumerate(titles):
        # Save file
        filename = safe_filename(title)
        out_path = corpus_dir / filename

        if os.path.exists(out_path):
            continue

        if i % 1000 == 0:
            print(f"[INFO] Processing {i}/{len(titles)}... saved={saved}, skipped={skipped}")

        # Fetch page
        page = wiki.page(title)
        if not page.exists():
            skipped += 1
            continue

        text = page.text
        if not text.strip():
            skipped += 1
            continue

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text)

        saved += 1

        # Avoid hitting API too fast
        time.sleep(SLEEP_BETWEEN_REQUESTS)

    print(f"\n[INFO] Corpus build completed.")
    print(f"[INFO] Saved pages: {saved}")
    print(f"[INFO] Skipped pages: {skipped}")
    print(f"[INFO] Files are in: {CORPUS_DIR}")


if __name__ == "__main__":
    build_corpus()
