"""
Build a large Turkish gazetteer using Wikipedia categories.
- Iterative traversal with explicit stack (no Python recursion)
- Tracks visited categories to avoid cycles
- Cleans titles (removes parentheses)
- Filters invalid items
- Saves a large gazetteer list

Requirements:
    pip install wikipedia-api
"""

import wikipediaapi
import re
from typing import Set


# -------------------------------------------------------------
# Wikipedia client
# -------------------------------------------------------------
wiki = wikipediaapi.Wikipedia(
    language="tr",
    user_agent="NLP-Normalization-Project/1.0 (contact: example@example.com)"
)


# -------------------------------------------------------------
# Title cleaning
# -------------------------------------------------------------
def clean_title(title: str) -> str:
    """
    Remove parentheses (e.g., 'Adana (il)' -> 'Adana')
    Remove leading/trailing whitespace
    """
    title = re.sub(r"\s*\(.*?\)\s*", "", title).strip()
    return title


# -------------------------------------------------------------
# Filtering heuristic
# -------------------------------------------------------------
def is_valid_title(title: str) -> bool:
    """
    Simple validity checks:
    - Must start with uppercase Turkish letter
    - Cannot be numeric
    - Cannot be empty
    """
    if not title:
        return False
    if title.isdigit():
        return False
    if not title[0].isalpha():
        return False
    if not title[0].isupper():
        return False
    return True


# -------------------------------------------------------------
# Iterative category collector (no recursion)
# -------------------------------------------------------------
def collect_category_recursive(category_name: str, limit: int = 50000) -> Set[str]:
    """
    Iteratively collects all pages inside a Wikipedia category.
    This includes subcategories and their pages.
    Uses an explicit stack and a visited set to avoid infinite recursion.
    """

    root = wiki.page("Kategori:" + category_name)
    collected: Set[str] = set()

    if not root.exists():
        print(f"[WARN] Category not found: {category_name}")
        return collected

    stack = [root]
    visited_categories: Set[str] = set()

    while stack and len(collected) < limit:
        page = stack.pop()

        # Avoid revisiting the same category
        if page.title in visited_categories:
            continue
        visited_categories.add(page.title)

        try:
            members = page.categorymembers
        except Exception as e:
            print(f"[WARN] Could not fetch categorymembers for {page.title}: {e}")
            continue

        for title, member in members.items():
            # Subcategory → push to stack
            if member.ns == wikipediaapi.Namespace.CATEGORY:
                if member.title not in visited_categories:
                    stack.append(member)

            # Main article
            elif member.ns == wikipediaapi.Namespace.MAIN:
                clean = clean_title(title)
                if is_valid_title(clean):
                    collected.add(clean)
                    if len(collected) >= limit:
                        break

    return collected


# -------------------------------------------------------------
# Categories to include
# -------------------------------------------------------------
TURKISH_CATEGORIES = [
    "Türk bilim insanları",
    "Türk müzisyenler",
    "Türk futbolcular",
    "Türk sinema oyuncuları",
    "Türk yazarlar",
    "Türk şarkıcılar",
    "Türk siyasetçiler",
    "Türk sporcular",
    "Türkiye'nin illeri",
    "Türkiye'deki şehirler",
    "Türkiye'deki üniversiteler",
    "Türkiye'deki futbol kulüpleri",
    "Türkiye'deki spor kulüpleri",
]


# -------------------------------------------------------------
# Build the big gazetteer
# -------------------------------------------------------------
def build_gazetteer(output_path: str = "tr_gazetteer_large.txt"):
    gazetteer = set()

    for cat in TURKISH_CATEGORIES:
        print(f"\n[INFO] Collecting category: {cat}")
        items = collect_category_recursive(cat, limit=100000)
        print(f"[INFO] Found {len(items)} items in category '{cat}'")
        gazetteer.update(items)

    print("\n[INFO] Total gazetteer size:", len(gazetteer))

    # Save
    with open(output_path, "w", encoding="utf-8") as f:
        for item in sorted(gazetteer):
            f.write(item + "\n")

    print(f"[INFO] Gazetteer saved to {output_path}")


# -------------------------------------------------------------
# Main
# -------------------------------------------------------------
if __name__ == "__main__":
    build_gazetteer()
