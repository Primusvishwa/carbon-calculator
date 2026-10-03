"""
find_ef.py - simple, deterministic text search over the emission_factors table.

COMMAND LINE (run from the folder that contains emission_factors.db):
    python find_ef.py "cotton cultivation"
    python find_ef.py "cotton cultivation" --stage raw_material_extraction
    python find_ef.py "HGV freight" --stage transport --unit tonne.km
    python find_ef.py "grid electricity" --geo "united kingdom"
    python find_ef.py "landfill" --top 10

FROM PYTHON (for testing):
    from find_ef import find_ef
    find_ef("HGV freight", stage="transport", unit="tonne.km")

Stage values: raw_material_extraction, intermediate_processing,
              manufacturing_assembly, transport, end_of_life
"""

import argparse
import re
import sqlite3

DB_PATH = "emission_factors.db"

# Filler words that carry no meaning for matching, so we ignore them.
# "waste", "sent", "disposal", "disposed" were added because "waste" matches
# dozens of unrelated waste rows and pushed the right row (e.g. Clothing / Landfill)
# out of the shortlist.
STOPWORDS = {"of", "the", "a", "an", "and", "or", "for", "to", "in", "on", "by", "with", "raw",
             "waste", "sent", "disposal", "disposed"}

# Everyday words mapped to the wording used in the database.
# If a search misses because of wording, add a pair here.
SYNONYMS = {
    # transport
    "truck": ["hgv"],
    "lorry": ["hgv"],
    "ship": ["cargo"],
    "shipping": ["cargo"],
    "sea": ["cargo"],
    # energy
    "power": ["electricity"],
    # textiles: the database calls finished textile products "clothing"
    "textile": ["clothing"],
    "garment": ["clothing"],
    "apparel": ["clothing"],
    "shirt": ["clothing"],
}


def clean_words(text):
    """Lowercase, split into words, drop filler words and single letters, trim plural 's'."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    result = []
    for w in words:
        if w in STOPWORDS or len(w) < 2:
            continue
        if len(w) > 3 and w.endswith("s"):
            w = w[:-1]
        result.append(w)
    return result


def word_pattern(word):
    """Match the word at the START of a word in the text.
    So 'landfill' matches 'landfilled', but 'tee' will not match 'steel'."""
    return re.compile(r"\b" + re.escape(word))


def data_warning(row):
    """Return a short caution for rows whose number needs care, or None."""
    only_co2 = (
        row["gas_basis"] == "raw_mass"
        and row["value_co2"] is not None
        and row["value_ch4"] is None
        and row["value_n2o"] is None
    )
    if only_co2:
        return "CO2 only - CH4 and N2O not included, so total is likely understated"
    return None


def study_year(row):
    """Textile Exchange rows store the underlying study's year in the notes
    (e.g. 'Pub. year: 2012'). Return it as text, or None if not present."""
    m = re.search(r"Pub\. year:\s*(\d{4})", row["notes"] or "")
    return m.group(1) if m else None


def find_ef(query, stage=None, geography=None, unit=None, top=5, show=True):
    """
    Search emission_factors for rows matching `query`.

    stage     : only rows with this lca_stage (exact value, see top of file)
    geography : only rows whose geography contains this text (e.g. "china")
    unit      : only rows whose unit contains this text (e.g. "tonne.km")
    top       : how many results to return/print
    show      : print results (True) or just return them (False)
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    words = clean_words(query)
    search_words = list(words)
    for w in words:
        search_words += SYNONYMS.get(w, [])
    patterns = {w: word_pattern(w) for w in set(search_words)}

    # Read from the view (has co2e_total) and join sources to get the source year.
    sql = """
        SELECT v.*, s.year AS source_year
        FROM v_emission_factors v
        LEFT JOIN sources s ON s.source_id = v.source_id
        WHERE 1 = 1
    """
    params = []
    if stage:
        sql += " AND v.lca_stage = ?"
        params.append(stage)
    if geography:
        sql += " AND LOWER(v.geography) LIKE ?"
        params.append(f"%{geography.lower()}%")
    if unit:
        sql += " AND LOWER(v.unit) LIKE ?"
        params.append(f"%{unit.lower()}%")
    rows = conn.execute(sql, params).fetchall()

    scored = []
    for row in rows:
        activity = (row["activity"] or "").lower()
        category = (row["category"] or "").lower()
        variant = (row["variant"] or "").lower()

        score = 0
        matched = set()
        for w in search_words:
            p = patterns[w]
            if p.search(activity):
                score += 3      # activity is the most important column
                matched.add(w)
            if p.search(category):
                score += 2
                matched.add(w)
            if p.search(variant):
                score += 1
                matched.add(w)

        # Bonus if the whole phrase appears in the activity text.
        if words and " ".join(words) in activity:
            score += 5

        # Bonus for covering more of the words the user typed (not synonyms).
        score += sum(1 for w in words if w in matched) * 2

        if score > 0:
            scored.append((score, row))

    # Highest score first; ties broken by ef_id so the order is stable.
    scored.sort(key=lambda pair: (-pair[0], pair[1]["ef_id"]))
    best = scored[:top]

    if show:
        filters = []
        if stage:
            filters.append(f"stage={stage}")
        if geography:
            filters.append(f"geo~{geography}")
        if unit:
            filters.append(f"unit~{unit}")
        print("=" * 72)
        print(f"Query: {query!r}   Filters: {', '.join(filters) or 'none'}")
        print(f"{len(scored)} rows matched, showing top {len(best)}")
        print("=" * 72)
        if not best:
            print("No matches found.")
            if filters:
                print("(Try removing a filter - the data may simply not cover that combination.)")
            print()
        for score, r in best:
            co2e = r["co2e_total"]
            co2e_text = f"{co2e:.4f}" if co2e is not None else "n/a"
            print(f"[score {score}]  ef_id {r['ef_id']}")
            print(f"   activity : {r['activity']}")
            print(f"   category : {r['category']}")
            print(f"   variant  : {r['variant']}")
            print(f"   unit     : {r['unit']}   co2e_total: {co2e_text}")
            print(f"   stage    : {r['lca_stage']}   geography: {r['geography']}")
            src = r["source_name"]
            if r["source_year"]:
                src += f" ({r['source_year']})"
            yr = study_year(r)
            if yr:
                src += f" - underlying study published {yr}"
            print(f"   source   : {src}")
            warn = data_warning(r)
            if warn:
                print(f"   WARNING  : {warn}")
        print()

    conn.close()
    return [dict(r) for _, r in best]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Search the emission_factors table.")
    parser.add_argument("query", help='what to search for, e.g. "cotton cultivation"')
    parser.add_argument("--stage", help="filter by lca_stage, e.g. transport")
    parser.add_argument("--geo", help='filter by geography text, e.g. "china"')
    parser.add_argument("--unit", help='filter by unit text, e.g. "tonne.km"')
    parser.add_argument("--top", type=int, default=5, help="number of results (default 5)")
    args = parser.parse_args()
    find_ef(args.query, stage=args.stage, geography=args.geo, unit=args.unit, top=args.top)