"""
run_pipeline.py - the whole carbon calculator in one go:

    search (find_ef.py)  ->  AI picks the row (ai_match.py)  ->  calculation (calculate_footprint.py)

You list your process steps with their quantities. For each step this script:
  - searches the database for candidate rows,
  - asks the AI to pick the right row (or say none fit), with one second chance using
    alternative search phrases if it says none fit,
  - then hands the chosen rows to the calculator, which does the arithmetic (no AI involved).

Put this file in the same folder as find_ef.py, ai_match.py, calculate_footprint.py and
emission_factors.db, make sure GEMINI_API_KEY is set, then run:
    python run_pipeline.py
It takes about two minutes (the free Gemini tier needs pauses between calls).
"""

import os
import sys
import time

from find_ef import find_ef
from ai_match import choose_match, suggest_queries
from calculate_footprint import (calculate_footprint, print_report, summarise,
                                 factor_denominator, normalise_unit, CalcError)

TOP_N = 10           # candidates the first search hands to the AI
RETRY_TOP_N = 5      # candidates each alternative search phrase adds on the second look
LOCAL_TOP_N = 5      # extra candidates taken only from the target country (see search() below)
PAUSE_SECONDS = 7    # pause between AI calls (free-tier rate limit)
TARGET_GEOGRAPHY = "India"   # where your supply chain is; change it, or set to None

# ---------------------------------------------------------------------------
# YOUR INPUT: one entry per process step.
#   step        : plain description of the step (this is what gets searched)
#   stage       : optional LCA stage filter, or None. One of: raw_material_extraction,
#                 intermediate_processing, manufacturing_assembly, transport, end_of_life
#   quantity    : how much (a number)
#   unit        : unit of that quantity (kg, g, tonne, kWh, MWh, ...)
#   distance_km : ONLY for freight steps - how far the goods travel
# The quantities and distances are PLACEHOLDERS - replace them with your real numbers.
# ---------------------------------------------------------------------------
STEPS = [
    {"step": "cotton fibre", "stage": "raw_material_extraction", "quantity": 0.18, "unit": "kg"},
    {"step": "HGV freight", "stage": "transport", "quantity": 0.18, "unit": "kg", "distance_km": 500},
    {"step": "sea freight by container ship", "stage": "transport",
     "quantity": 0.18, "unit": "kg", "distance_km": 10000},
    {"step": "grid electricity India", "stage": None, "quantity": 1.5, "unit": "kWh"},
    {"step": "t-shirt landfill", "stage": "end_of_life", "quantity": 0.18, "unit": "kg"},
    {"step": "cotton yarn spinning", "stage": None, "quantity": 0.18, "unit": "kg"},
    {"step": "woven fabric weaving", "stage": None, "quantity": 0.18, "unit": "kg"},
    {"step": "reactive dyeing of cotton fabric", "stage": None, "quantity": 0.18, "unit": "kg"},
]

_ai_calls = 0


def throttle():
    """Wait a few seconds before every AI call except the first one."""
    global _ai_calls
    if _ai_calls > 0:
        time.sleep(PAUSE_SECONDS)
    _ai_calls += 1


def unit_fits(row, unit_kind):
    """Does this database row have the kind of unit the step needs?
    unit_kind 'tonne.km' = freight carrying a weight; 'distance' = per km / per mile of vehicle travel."""
    try:
        denominator = normalise_unit(factor_denominator(row["co2e_unit"]))
    except CalcError:
        return False
    if unit_kind == "tonne.km":
        return denominator == "tonne.km"
    if unit_kind == "distance":
        return denominator in ("km", "miles")
    return True


def search(text, stage, top, unit_kind=None):
    """Search the database. Also adds the best rows from the TARGET country, so a local factor
    (e.g. the India grid row) is never crowded out by dozens of rows from elsewhere.
    If unit_kind is given, only rows whose unit fits are kept (plain code, not the AI).
    Returns (rows, how_many_local_rows_were_added)."""
    fetch = top if not unit_kind else max(top * 8, 80)
    rows = find_ef(text, stage=stage, top=fetch, show=False)
    if unit_kind:
        rows = [r for r in rows if unit_fits(r, unit_kind)][:top]
    added = 0
    if TARGET_GEOGRAPHY:
        have = {int(r["ef_id"]) for r in rows}
        local_fetch = LOCAL_TOP_N if not unit_kind else max(LOCAL_TOP_N * 8, 40)
        local = find_ef(text, stage=stage, geography=TARGET_GEOGRAPHY, top=local_fetch, show=False)
        if unit_kind:
            local = [r for r in local if unit_fits(r, unit_kind)][:LOCAL_TOP_N]
        for r in local:
            if int(r["ef_id"]) not in have:
                rows.append(r)
                have.add(int(r["ef_id"]))
                added += 1
    return rows, added


def match_step(step_text, stage, unit_kind=None, unit_note=None):
    """Find the emission-factor row for one step.
    unit_kind / unit_note (optional): which kind of unit the step needs, and a plain-language note
    for the AI. Used for freight: 'tonne.km' (carries a weight) or 'distance' (trip distance only).
    Returns a dict with: ef_id (or None), confidence, reasoning, notes (what happened),
    row (the chosen database row, or None), pool (every candidate row the search looked at,
    used to offer stand-ins when there is no match) and problem (None, or text if the step
    could NOT be properly checked - e.g. the AI call failed - which is different from 'no match')."""
    out = {"ef_id": None, "confidence": None, "reasoning": "", "notes": [], "row": None,
           "problem": None, "pool": []}

    ai_text = step_text if not unit_note else f"{step_text}  [Note: {unit_note}]"

    # ---- search ----
    candidates, local_added = search(step_text, stage, TOP_N, unit_kind)
    note = f"Search found {len(candidates)} candidate(s)."
    if unit_kind:
        kind_text = "per tonne.km" if unit_kind == "tonne.km" else "per km / per mile of vehicle travel"
        note = f"Search found {len(candidates)} candidate(s), keeping only rows whose unit is {kind_text}."
    if local_added:
        note += f" (Includes {local_added} extra from {TARGET_GEOGRAPHY}, so local data is not crowded out.)"
    out["notes"].append(note)
    out["pool"] = list(candidates)

    # ---- AI first look ----
    first_reasoning = ""
    if candidates:
        throttle()
        r = choose_match(ai_text, candidates, stage=stage, geography=TARGET_GEOGRAPHY)
        if r.get("error"):
            out["problem"] = f"the AI matching call failed, so this step was NOT checked: {r['reasoning']}"
            return out
        if r["match"] is not None:
            out.update(ef_id=r["match"], confidence=r["confidence"], reasoning=r["reasoning"],
                       row=next(c for c in candidates if int(c["ef_id"]) == r["match"]))
            return out
        first_reasoning = r["reasoning"]
        out["notes"].append("AI found no match on the first look.")
    else:
        out["notes"].append("AI was not called (nothing to choose from).")

    # ---- second chance: alternative search phrases ----
    throttle()
    phrases = suggest_queries(step_text, stage=stage)
    if not phrases:
        out["problem"] = ("no match on the first look, and the second-chance search could not run "
                          "(the AI call failed or gave no phrases), so this step was NOT fully checked")
        return out
    seen = {int(c["ef_id"]) for c in candidates}
    extra = []
    for phrase in phrases:
        rows, _ = search(phrase, stage, RETRY_TOP_N, unit_kind)
        for row in rows:
            if int(row["ef_id"]) not in seen:
                seen.add(int(row["ef_id"]))
                extra.append(row)
    out["notes"].append(f"Second chance: AI suggested {phrases}; {len(extra)} new candidate(s) found.")
    if not extra:
        out["reasoning"] = first_reasoning or "no candidates found, even with alternative search phrases"
        return out

    throttle()
    pool = candidates + extra
    out["pool"] = pool
    r2 = choose_match(ai_text, pool, stage=stage, geography=TARGET_GEOGRAPHY)
    if r2.get("error"):
        out["problem"] = f"the AI matching call failed on the second look, so this step was NOT fully checked: {r2['reasoning']}"
        return out
    out["reasoning"] = r2["reasoning"]
    if r2["match"] is not None:
        out.update(ef_id=r2["match"], confidence=r2["confidence"],
                   row=next(c for c in pool if int(c["ef_id"]) == r2["match"]))
    return out


def main():
    sys.stdout.reconfigure(errors="replace")
    if not os.environ.get("GEMINI_API_KEY"):
        sys.exit("No API key found. Set the GEMINI_API_KEY environment variable first.")

    print("=" * 78)
    print("PART 1 - MATCHING (search + AI)")
    print("=" * 78)
    calc_steps = []
    problems = {}
    for i, s in enumerate(STEPS, 1):
        print(f"STEP {i}: {s['step']}    (stage filter: {s.get('stage') or 'none'})", flush=True)
        # Freight that carries a weight over a distance must use per-tonne.km factors.
        kind = "tonne.km" if "distance_km" in s else None
        unit_note = "freight carrying a weight: the factor must be per tonne.km" if kind else None
        m = match_step(s["step"], s.get("stage"), kind, unit_note)
        for note in m["notes"]:
            print(f"   {note}")
        if m["problem"]:
            print(f"   PROBLEM: {m['problem']}")
        elif m["ef_id"] is None:
            print("   DECISION: NO MATCH - needs external source")
            print(f"   AI reasoning: {m['reasoning']}")
        else:
            row = m["row"]
            print(f"   DECISION: MATCHED ef_id {m['ef_id']} - {row['activity']} "
                  f"({row['geography']}), confidence: {m['confidence']}")
            print(f"   AI reasoning: {m['reasoning']}")
        print()

        item = {"step": s["step"], "quantity": s["quantity"], "unit": s["unit"],
                "ef_id": m["ef_id"], "confidence": m["confidence"]}
        if "distance_km" in s:
            item["distance_km"] = s["distance_km"]
        calc_steps.append(item)
        if m["problem"]:
            problems[i - 1] = m["problem"]

    print()
    print("PART 2 - CALCULATION (plain arithmetic, no AI)")
    results, _ = calculate_footprint(calc_steps)
    # A step the AI could not properly check is NOT a genuine "no match": say so in the gap list.
    for idx, why in problems.items():
        results[idx]["status"] = "cannot_calculate"
        results[idx]["reason"] = why
    summary = summarise(results)
    print_report(results, summary)


if __name__ == "__main__":
    main()