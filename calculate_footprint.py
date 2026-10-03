"""
calculate_footprint.py - the calculation step of the carbon calculator.

NO AI is used in this file. It is plain arithmetic:
    contribution = (quantity converted to the factor's unit) x (the factor's CO2e value)

For every process step it uses the ef_id that ai_match.py already decided, looks that row up
in the database view v_emission_factors (which gives the safe combined CO2e figure, co2e_total),
converts units where needed, and multiplies.

HOW STEPS ARE TREATED (a deliberate design rule):
  - matched, confidence high or medium -> counted in BOTH totals
  - matched, confidence low            -> a real factor used as an imperfect stand-in (proxy):
                                          counted in the GRAND total only, and flagged everywhere
  - matched, but the factor covers CO2 only (no methane / nitrous oxide) -> by default just a
                                          NOTE on the step; it still counts as a normal match.
                                          If you switch DEMOTE_CO2_ONLY on, such a step is counted
                                          in the GRAND total only (except the categories listed in
                                          CO2_ONLY_OK_CATEGORIES, e.g. electricity).
  - no match (ef_id is None)           -> excluded from every total, listed as an OPEN GAP.
                                          Missing data is never treated as zero.
  - matched but cannot be calculated   -> (e.g. units cannot be converted) excluded from every
                                          total and listed as an OPEN GAP with the reason.

Put this file next to find_ef.py and emission_factors.db, then run:
    python calculate_footprint.py
No API key is needed.
"""

import math
import sqlite3
import sys

from find_ef import DB_PATH, data_warning

# ---------------------------------------------------------------------------
# YOUR INPUT: one entry per process step.
#   step        : name of the step
#   quantity    : how much of it (a number)
#   unit        : the unit of that quantity (kg, g, tonne, kWh, MWh, ...)
#   distance_km : ONLY for freight steps - how far the goods travel
#   ef_id       : the row ai_match.py chose, or None if it said "no match"
#   confidence  : "high", "medium" or "low" from ai_match.py (None if no match)
# The ef_id and confidence values below are copied from your last test_pipeline run.
# The quantities and distances are PLACEHOLDERS - replace them with your real numbers.
# ---------------------------------------------------------------------------
STEPS = [
    {"step": "cotton fibre", "quantity": 0.18, "unit": "kg",
     "ef_id": 871, "confidence": "medium"},
    {"step": "HGV freight", "quantity": 0.18, "unit": "kg", "distance_km": 500,
     "ef_id": 317, "confidence": "low"},
    {"step": "sea freight by container ship", "quantity": 0.18, "unit": "kg", "distance_km": 10000,
     "ef_id": 469, "confidence": "medium"},
    {"step": "grid electricity India", "quantity": 1.5, "unit": "kWh",
     "ef_id": 922, "confidence": "high"},
    {"step": "t-shirt landfill", "quantity": 0.18, "unit": "kg",
     "ef_id": 518, "confidence": "low"},
    {"step": "cotton yarn spinning", "quantity": 0.18, "unit": "kg",
     "ef_id": None, "confidence": None},
    {"step": "woven fabric weaving", "quantity": 0.18, "unit": "kg",
     "ef_id": None, "confidence": None},
    {"step": "reactive dyeing of cotton fabric", "quantity": 0.18, "unit": "kg",
     "ef_id": None, "confidence": None},
]

# ---------------------------------------------------------------------------
# CO2-only factors. Some database rows count only CO2 (no methane, no nitrous oxide). This is a
# property of the database row, not of your question.
# DEMOTE_CO2_ONLY = False (default): such a row is used like any other matched row, and the step just
#                    carries a note saying the factor is CO2 only.
# DEMOTE_CO2_ONLY = True : such a step is counted in the grand total only (like a proxy), except the
#                    categories listed below.
# ---------------------------------------------------------------------------
DEMOTE_CO2_ONLY = False
CO2_ONLY_OK_CATEGORIES = {"electricity"}   # lower-case category names where CO2 alone is fine

# ---------------------------------------------------------------------------
# Unit handling. Only conversions listed here are allowed. Anything else stops that
# step with a clear message instead of guessing.
# ---------------------------------------------------------------------------
UNIT_ALIASES = {
    "g": "g", "gram": "g", "grams": "g",
    "kg": "kg", "kgs": "kg", "kilogram": "kg", "kilograms": "kg", "kg product": "kg",
    "t": "tonne", "tonne": "tonne", "tonnes": "tonne", "metric ton": "tonne", "metric tons": "tonne",
    "short_ton": "short_ton", "short ton": "short_ton", "short tons": "short_ton",
    "kwh": "kwh", "mwh": "mwh", "mj": "mj",
    "km": "km", "mile": "miles", "miles": "miles",
    "tonne.km": "tonne.km", "tonnes.km": "tonne.km", "tonne-km": "tonne.km", "tkm": "tonne.km",
}
MASS_IN_KG = {"g": 0.001, "kg": 1.0, "tonne": 1000.0, "short_ton": 907.18474}
ENERGY_IN_KWH = {"kwh": 1.0, "mwh": 1000.0, "mj": 1.0 / 3.6}
DISTANCE_IN_KM = {"km": 1.0, "miles": 1.609344}


class CalcError(Exception):
    """Raised when a step cannot be calculated safely. The message explains why."""


def normalise_unit(unit):
    text = str(unit).strip().lower()
    return UNIT_ALIASES.get(text, text)


def num(x):
    return f"{x:.6g}"


def factor_denominator(co2e_unit):
    """'kg CO2e per tonnes' -> 'tonnes'.  'kg CO2 per kg product' -> 'kg product'.
    Also checks that the top part really is kg of CO2 / CO2e."""
    text = str(co2e_unit)
    if " per " not in text:
        raise CalcError(f"cannot read the factor's unit: '{text}'")
    top, bottom = text.split(" per ", 1)
    if not top.strip().lower().startswith("kg co2"):
        raise CalcError(f"factor is not expressed in kg CO2e: '{text}'")
    return bottom.strip()


def convert_quantity(quantity, step_unit, factor_unit, distance_km=None):
    """Convert the step's quantity into the unit the emission factor is 'per'.
    Returns (converted_quantity, plain-language note). Raises CalcError if not possible."""
    s = normalise_unit(step_unit)
    f = normalise_unit(factor_unit)

    # Freight: the factor is per tonne.km, so we need mass AND distance.
    if f == "tonne.km":
        if s not in MASS_IN_KG:
            raise CalcError(f"freight factor is per tonne.km but the step quantity is in "
                            f"'{step_unit}' (need a weight such as kg or tonne)")
        if distance_km is None:
            raise CalcError("freight factor is per tonne.km, so this step needs a 'distance_km' value")
        if distance_km < 0:
            raise CalcError("distance_km cannot be negative")
        tonnes = quantity * MASS_IN_KG[s] / 1000.0
        value = tonnes * distance_km
        return value, (f"{num(quantity)} {step_unit} = {num(tonnes)} tonne; "
                       f"x {num(distance_km)} km = {num(value)} tonne.km")

    # Same kind of unit on both sides (weight, energy or distance): scale between them.
    for table in (MASS_IN_KG, ENERGY_IN_KWH, DISTANCE_IN_KM):
        if s in table and f in table:
            value = quantity * table[s] / table[f]
            if s == f:
                return value, "no conversion needed"
            return value, f"{num(quantity)} {step_unit} = {num(value)} {factor_unit}"

    # Identical unit names that we have no table for (e.g. litres and litres).
    if s == f:
        return quantity, "no conversion needed"

    raise CalcError(f"cannot convert '{step_unit}' to the factor's unit '{factor_unit}'")


def calculate_step(conn, step):
    """Work out one step. Never raises: problems become an excluded 'cannot_calculate' result."""
    res = {
        "step": step["step"], "quantity": step["quantity"], "unit": step["unit"],
        "ef_id": step.get("ef_id"), "confidence": step.get("confidence"),
        "status": None, "contribution": None, "reason": None,
        "activity": None, "variant": None, "geography": None, "source": None,
        "factor_value": None, "factor_unit": None, "converted": None,
        "conversion_note": None, "notes": [], "proxy_reason": None,
    }

    # ---- Situation 2: no match at all -> excluded, never zero ----
    if res["ef_id"] is None:
        res["status"] = "no_match"
        res["reason"] = "no emission factor in the database matched this step"
        return res

    row = conn.execute("SELECT * FROM v_emission_factors WHERE ef_id = ?",
                       (res["ef_id"],)).fetchone()
    if row is None:
        res["status"] = "cannot_calculate"
        res["reason"] = f"ef_id {res['ef_id']} was not found in the database"
        return res

    res["activity"], res["variant"] = row["activity"], row["variant"]
    res["geography"], res["source"] = row["geography"], row["source_name"]
    res["factor_value"], res["factor_unit"] = row["co2e_total"], row["co2e_unit"]
    warning = data_warning(row)
    if warning:
        res["notes"].append(warning)

    try:
        q = step["quantity"]
        if isinstance(q, bool) or not isinstance(q, (int, float)) or not math.isfinite(q) or q < 0:
            raise CalcError(f"quantity must be a number of zero or more (got {q!r})")
        if row["co2e_total"] is None:
            raise CalcError("this database row has no CO2e value")
        denominator = factor_denominator(row["co2e_unit"])
        converted, note = convert_quantity(q, step["unit"], denominator, step.get("distance_km"))
        if step.get("distance_km") is not None and normalise_unit(denominator) != "tonne.km":
            res["notes"].append("distance_km was given but this factor is not per tonne.km, "
                                "so the distance was not used")
    except CalcError as e:
        res["status"] = "cannot_calculate"
        res["reason"] = f"matched, but could not calculate: {e}"
        return res

    res["converted"], res["conversion_note"] = converted, note
    res["contribution"] = converted * row["co2e_total"]

    # ---- Situation 1: matched. High/medium = clean; low (or unknown) = proxy ----
    # A CO2-only factor is incomplete data (see DEMOTE_CO2_ONLY above) and also becomes a proxy.
    confidence = res["confidence"]
    partial = (DEMOTE_CO2_ONLY and bool(warning)
               and str(row["category"] or "").strip().lower() not in CO2_ONLY_OK_CATEGORIES)
    if confidence in ("high", "medium") and not partial:
        res["status"] = "clean"
    else:
        res["status"] = "proxy"
        if confidence in ("high", "medium"):
            res["proxy_reason"] = "co2_only"
            res["notes"].append("Because this factor covers CO2 only, it is counted in the grand total "
                                "only (not in the high/medium subtotal).")
        elif confidence != "low":
            res["notes"].append(f"confidence was {confidence!r} (not high/medium/low), "
                                "so it is treated as low")
    return res


def summarise(results):
    clean = [r["contribution"] for r in results if r["status"] == "clean"]
    proxy = [r["contribution"] for r in results if r["status"] == "proxy"]
    gaps = [r for r in results if r["status"] in ("no_match", "cannot_calculate")]
    return {
        "clean_total": math.fsum(clean), "clean_count": len(clean),
        "proxy_total": math.fsum(proxy), "proxy_count": len(proxy),
        "grand_total": math.fsum(clean + proxy),
        "gaps": gaps, "step_count": len(results),
    }


def calculate_footprint(steps):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        results = [calculate_step(conn, s) for s in steps]
    finally:
        conn.close()
    return results, summarise(results)


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------
def status_label(r):
    if r["status"] == "clean":
        return f"included ({r['confidence']} confidence)"
    if r["status"] == "proxy":
        return "CO2-ONLY FACTOR" if r.get("proxy_reason") == "co2_only" else "LOW-CONFIDENCE PROXY"
    if r["status"] == "no_match":
        return "EXCLUDED - no match"
    return "EXCLUDED - could not calculate"


def print_report(results, summary):
    print("=" * 78)
    print("CARBON FOOTPRINT CALCULATION  (all figures in kg CO2e)")
    print("=" * 78)

    for i, r in enumerate(results, 1):
        print(f"STEP {i}: {r['step']}    ({num(r['quantity'])} {r['unit']})")
        if r["status"] == "no_match":
            print("   EXCLUDED - NO MATCH. No emission factor exists in the database for this step.")
            print("   Not counted as zero. Listed under OPEN GAPS below.\n")
            continue
        if r["activity"]:
            print(f"   Matched ef_id {r['ef_id']}: {r['activity']}  |  geography: {r['geography']}"
                  f"  |  confidence: {r['confidence']}")
            print(f"   Factor : {num(r['factor_value'])} {r['factor_unit']}   [source: {r['source']}]")
        if r["status"] == "cannot_calculate":
            print(f"   EXCLUDED - {r['reason']}")
            print("   Not counted as zero. Listed under OPEN GAPS below.")
        else:
            print(f"   Convert: {r['conversion_note']}")
            print(f"   Result : {num(r['converted'])} x {num(r['factor_value'])} = "
                  f"{r['contribution']:.4f} kg CO2e")
            if r["status"] == "proxy" and r.get("proxy_reason") == "co2_only":
                print("   FLAG   : CO2-ONLY FACTOR (no methane or nitrous oxide), so this "
                      "probably understates.")
                print("            Included in the GRAND total only, not in the high/medium subtotal.")
            elif r["status"] == "proxy":
                print("   FLAG   : LOW-CONFIDENCE PROXY MATCH - real data, but an imperfect stand-in "
                      "(e.g. different country).")
                print("            Included in the GRAND total only, not in the high/medium subtotal.")
        for note in r["notes"]:
            print(f"   NOTE   : {note}")
        print()

    print("=" * 78)
    print("AT A GLANCE")
    print("=" * 78)
    for r in results:
        value = f"{r['contribution']:.4f}" if r["contribution"] is not None else "--"
        print(f"  {r['step']:<36} {status_label(r):<32} {value:>10}")

    n_gaps = len(summary["gaps"])
    covered = summary["step_count"] - n_gaps
    print()
    print("=" * 78)
    print("TOTALS (kg CO2e)")
    print("=" * 78)
    if covered == 0:
        print("  NO TOTAL: not a single step could be calculated.")
        print("  This is NOT zero - every step is still missing data (see OPEN GAPS below).")
    else:
        proxy_label = ("+ Proxy, low-confidence or CO2-only steps" if DEMOTE_CO2_ONLY
                       else "+ Proxy and low-confidence steps")
        print(f"  {'High/medium-confidence subtotal':<44}: {summary['clean_total']:.4f}"
              f"   ({summary['clean_count']} steps)")
        print(f"  {proxy_label:<44}: {summary['proxy_total']:.4f}"
              f"   ({summary['proxy_count']} steps)")
        print(f"  {'Grand total incl. all of the above':<44}: {summary['grand_total']:.4f}")
        if n_gaps:
            print()
            print(f"  WARNING: these totals cover only {covered} of {summary['step_count']} steps.")
            print(f"  The {n_gaps} gap step(s) below are NOT included (they are missing, not zero),")
            print("  so both totals understate the real footprint.")

    print()
    print("=" * 78)
    print(f"OPEN GAPS - NOT IN ANY TOTAL, STILL NEED SOURCING ({n_gaps})")
    print("=" * 78)
    if not n_gaps:
        print("  None - every step was calculated.")
    for r in summary["gaps"]:
        print(f"  - {r['step']} ({num(r['quantity'])} {r['unit']}): {r['reason']}")


def main():
    # Stop odd characters in database text (e.g. long dashes) from crashing the printout on Windows.
    sys.stdout.reconfigure(errors="replace")
    results, summary = calculate_footprint(STEPS)
    print_report(results, summary)


if __name__ == "__main__":
    main()