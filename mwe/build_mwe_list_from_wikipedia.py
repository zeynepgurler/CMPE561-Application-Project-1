import requests
from typing import Set, List

API_URL = "https://en.wiktionary.org/w/api.php"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; NLPProjectBot/1.0; +https://example.com)"
}


def fetch_members_recursive(category: str, visited=None) -> Set[str]:
    if visited is None:
        visited = set()

    if category in visited:
        return set()

    visited.add(category)
    result = set()

    cmcontinue = None
    while True:
        params = {
            "action": "query",
            "format": "json",
            "list": "categorymembers",
            "cmtitle": category,
            "cmlimit": "500",
            "cmtype": "page|subcat",
        }
        if cmcontinue:
            params["cmcontinue"] = cmcontinue

        r = requests.get(API_URL, params=params, headers=HEADERS, timeout=20)
        r.raise_for_status()
        data = r.json()

        for item in data["query"]["categorymembers"]:
            title = item["title"]
            ns = item["ns"]

            # ns=14 → subcategory
            if ns == 14:
                subcat = "Category:" + title.split("Category:", 1)[-1]
                result |= fetch_members_recursive(subcat, visited)

            # ns=0 → main page
            elif ns == 0:
                if " " in title:  # multiword süzgeci
                    result.add(title)

        if "continue" in data:
            cmcontinue = data["continue"]["cmcontinue"]
        else:
            break

    return result


def main():
    ROOT = "Category:Turkish multiword terms"

    print(f"Fetching from: {ROOT}")
    mwes = fetch_members_recursive(ROOT)
    print("Toplam:", len(mwes))

    out_file = "wiktionary_turkish_mwe.txt"
    with open(out_file, "w", encoding="utf-8") as f:
        for m in sorted(mwes):
            f.write(m + "\n")

    print("Kaydedildi:", out_file)


if __name__ == "__main__":
    main()
