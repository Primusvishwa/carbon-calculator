"""
engine.py - the carbon calculator as plain functions (no printing, no keyboard questions),
so the web app (app.py) can drive it one stage at a time.

It re-uses the same pieces as the command-line tool: find_ef.py (search), ai_match.py (the AI
picks a row), calculate_footprint.py (plain arithmetic) and the helpers in ask.py.

Safety limits for a public app live here and in app.py:
  - questions are limited in length and in number of steps
  - AI calls are paced so the free Gemini tier is not hammered
  - only one question is matched at a time (the matching code keeps a shared 'where is this
    supply chain' setting, so two at once could mix up)
"""

import csv
import datetime
import html
import io
import os
import re
import sqlite3
import threading
import time

import ask
import calculate_footprint as cf
import find_ef
import run_pipeline
from calculate_footprint import (CalcError, calculate_footprint, convert_quantity,
                                 factor_denominator, num, summarise)

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
DB_FILE = os.path.join(HERE, "emission_factors.db")
find_ef.DB_PATH = DB_FILE      # make sure every module finds the database, wherever the app runs
cf.DB_PATH = DB_FILE

MAX_STEPS = 6                  # most steps one question may contain
MAX_QUESTION_CHARS = 500
AI_MIN_GAP_SECONDS = 6.0       # the free Gemini tier allows roughly 10 AI calls a minute
DEFAULT_GEOGRAPHY = "India"

DISTANCE_NOTE = ("trip distance only, no weight: the factor must be per vehicle-km "
                 "(per km or per mile of vehicle travel), not per tonne.km")
WEIGHT_NOTE = "freight carrying a weight: the factor must be per tonne.km"

# ---------------------------------------------------------------------------
# Pacing of AI calls (shared by everybody using the app)
# ---------------------------------------------------------------------------
_gap_lock = threading.Lock()
_last_call = [0.0]
ENGINE_LOCK = threading.RLock()


def _paced_throttle():
    with _gap_lock:
        wait = _last_call[0] + AI_MIN_GAP_SECONDS - time.time()
        if wait > 0:
            time.sleep(wait)
        _last_call[0] = time.time()


run_pipeline.throttle = _paced_throttle   # replaces the command-line version of the pause


def scrub(text, limit=200):
    """Make an error message safe to show: hide the API key if it ever appears; keep it short."""
    text = str(text)
    key = os.environ.get("GEMINI_API_KEY", "")
    if key:
        text = text.replace(key, "[hidden]")
    return text if len(text) <= limit else text[: limit - 3] + "..."


# ---------------------------------------------------------------------------
# Stage 1: read the question
# ---------------------------------------------------------------------------
def parse_question(question, default_geo=DEFAULT_GEOGRAPHY):
    """Turn the question into steps. Returns {"steps": [...], "problem": None or text}.
    Distances are NOT final here: the person confirms or edits them next."""
    question = (question or "").strip()
    if not question:
        return {"steps": None, "problem": "Please type a question first."}
    if len(question) > MAX_QUESTION_CHARS:
        return {"steps": None, "problem": f"Please keep the question under {MAX_QUESTION_CHARS} characters."}

    data, err = ask.parse_question(question)
    if err:
        return {"steps": None, "problem": "The AI could not read your question right now "
                                          f"({scrub(err)}). Please try again in a minute."}
    parsed = data.get("steps") or []
    if not parsed:
        return {"steps": None,
                "problem": data.get("clarification") or "I could not tell what to calculate. Could you rephrase?"}
    if len(parsed) > MAX_STEPS:
        return {"steps": None, "problem": f"That question has more than {MAX_STEPS} steps. Please ask in smaller parts."}

    qnums = ask.numbers_in(question)
    steps = []
    for n, p in enumerate(parsed, 1):
        label = str(p.get("search_text") or "").strip()
        if not label:
            return {"steps": None, "problem": "I could not work out what activity you mean. Could you rephrase?"}
        q, u = p.get("quantity"), p.get("unit")
        is_freight = bool(p.get("is_freight"))
        basis = p.get("freight_basis")
        if is_freight and basis not in ("mass_distance", "distance_only"):
            basis = "mass_distance" if (q is not None and u) else "distance_only"
        distance_only = is_freight and basis == "distance_only"
        stage = p.get("stage") if p.get("stage") in ask.STAGES else None

        step = {"label": label, "stage": stage, "geography": p.get("geography") or default_geo,
                "is_freight": is_freight, "distance_only": distance_only,
                "quantity": None, "unit": None,
                "route_from": p.get("route_from"), "route_to": p.get("route_to"),
                "distance_km": None, "distance_how": None}

        if not distance_only:
            if q is None or not u:
                return {"steps": None, "problem": f"Step {n} ({label}): I could not find a quantity and unit. "
                                                  "Please include one, e.g. '100 kg' or '250 kWh'."}
            if not ask.appears_in(q, qnums):
                return {"steps": None, "problem": f"Step {n} ({label}): the AI read the quantity as {num(q)}, "
                                                  "but that number is not in your question. "
                                                  "Please write the quantity in digits."}
            step["quantity"], step["unit"] = float(q), str(u).strip()

        if is_freight:
            step["stage"] = "transport"
            stated = p.get("distance_stated")
            if stated is not None:
                if not ask.appears_in(stated, qnums):
                    return {"steps": None, "problem": f"Step {n} ({label}): the AI read a distance of {num(stated)} "
                                                      "but that number is not in your question. "
                                                      "Please write the distance in digits."}
                unit = str(p.get("distance_unit") or "km").strip().lower()
                if unit in ("km", "kms", "kilometre", "kilometres", "kilometer", "kilometers"):
                    step["distance_km"], step["distance_how"] = float(stated), "stated in your question"
                elif unit in ("mile", "miles", "mi"):
                    step["distance_km"] = float(stated) * 1.609344
                    step["distance_how"] = f"stated in your question ({num(stated)} miles, converted to km)"
                else:
                    return {"steps": None, "problem": f"Step {n} ({label}): distance unit '{unit}' not understood. "
                                                      "Use km or miles."}
            else:
                est = p.get("distance_km_estimate")
                if est is not None and est > 0 and p.get("route_from") and p.get("route_to"):
                    step["distance_km"], step["distance_how"] = float(est), "AI estimate, unverified"
                else:
                    step["distance_how"] = "not given - please enter it"
        steps.append(step)
    return {"steps": steps, "problem": None}


def build_run_steps(parsed_steps, km_values):
    """Combine the parsed steps with the distances the person confirmed.
    km_values: {step index: km}. Returns (run_steps, problem)."""
    run = []
    for i, p in enumerate(parsed_steps):
        s = {"label": p["label"], "stage": p["stage"], "geography": p["geography"],
             "distance_note": None, "unit_kind": None, "unit_note": None}
        if p["is_freight"]:
            km = km_values.get(i)
            if km is None or not (km > 0):
                return None, f"Step {i + 1} ({p['label']}): please enter a distance in km."
            default = p["distance_km"]
            if default is not None and abs(km - default) < 1e-9:
                how = p["distance_how"]
                if how and how.startswith("AI estimate"):
                    how = "AI estimate, unverified (accepted by you)"
            else:
                how = "entered by you"
            if p["distance_only"]:
                s.update(quantity=float(km), unit="km", trip_km=float(km), distance_note=how,
                         unit_kind="distance", unit_note=DISTANCE_NOTE)
            else:
                s.update(quantity=p["quantity"], unit=p["unit"], distance_km=float(km), trip_km=float(km),
                         distance_note=how, unit_kind="tonne.km", unit_note=WEIGHT_NOTE)
        else:
            s.update(quantity=p["quantity"], unit=p["unit"])
        run.append(s)
    return run, None


# ---------------------------------------------------------------------------
# Stage 2: find the emission factor for every step (search + AI)
# ---------------------------------------------------------------------------
def match_all(run_steps, progress=None):
    """Returns one match dict per step (see run_pipeline.match_step). Never raises."""
    matches = []
    with ENGINE_LOCK:
        for i, s in enumerate(run_steps):
            if progress:
                progress(i, s)
            run_pipeline.TARGET_GEOGRAPHY = s["geography"]
            try:
                m = run_pipeline.match_step(s["label"], s["stage"], s.get("unit_kind"), s.get("unit_note"))
            except Exception as e:      # never let one step crash the whole page
                m = {"ef_id": None, "confidence": None, "reasoning": "", "notes": [], "row": None,
                     "pool": [], "problem": f"unexpected error while matching this step: {scrub(e)}"}
            matches.append(m)
    return matches


def needs_stand_in(m):
    """True if the database has no exact factor for this step (and nothing went wrong)."""
    return m["ef_id"] is None and not m["problem"]


def stand_in_options(step, pool):
    """The closest stand-ins the database has, explained by the AI (the AI sees no numbers).
    Returns {"options": [...], "message": None or text}."""
    with ENGINE_LOCK:
        run_pipeline.TARGET_GEOGRAPHY = step["geography"]
        rows = ask.usable_rows(ask.widen_pool(step, pool), step)
        if not rows:
            return {"options": [], "message": "None of the rows the search found are in a unit that fits this step."}
        options, err = ask.suggest_proxy_options(step, rows)
    if err:
        return {"options": [], "message": f"Could not get stand-in suggestions right now ({scrub(err)})."}
    if not options:
        return {"options": [], "message": "The AI sees no reasonable stand-in among the rows found."}
    by_id = {int(r["ef_id"]): r for r in rows}
    out = []
    for o in options:
        r = by_id[o["ef_id"]]
        converted, _ = convert_quantity(step["quantity"], step["unit"],
                                        factor_denominator(r["co2e_unit"]), step.get("distance_km"))
        out.append({
            "ef_id": int(r["ef_id"]),
            "name": " | ".join(str(x) for x in (r["activity"], r["category"], r["variant"]) if x),
            "short_name": str(r["activity"]) + (f" - {r['variant']}" if r["variant"] else ""),
            "geography": r["geography"], "factor_value": r["co2e_total"], "factor_unit": r["co2e_unit"],
            "would_give": converted * r["co2e_total"],
            "why_close": o["why_close"], "how_it_differs": o["how_it_differs"], "likely_bias": o["likely_bias"],
        })
    return {"options": out, "message": None}


# ---------------------------------------------------------------------------
# Stage 3: arithmetic, plus everything needed to show where each factor came from
# ---------------------------------------------------------------------------
def factor_details(ef_id):
    """Everything the database knows about one factor, including its source record."""
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            """SELECT v.ef_id, v.activity, v.category, v.variant, v.unit, v.geography, v.lca_stage,
                      v.co2e_total, v.co2e_unit, v.source_id, v.source_sheet, v.source_row_ref,
                      v.notes, v.methodology, v.gas_basis, v.value_co2, v.value_ch4, v.value_n2o,
                      s.name AS source_name, s.publisher AS publisher, s.version AS source_version,
                      s.year AS source_year, s.license_type AS license_type, s.url AS source_url
               FROM v_emission_factors v LEFT JOIN sources s ON s.source_id = v.source_id
               WHERE v.ef_id = ?""", (ef_id,)).fetchone()
    finally:
        conn.close()
    if row is None:
        return None
    d = dict(row)
    m = re.search(r"Pub\. year:\s*(\d{4})", d.get("notes") or "")
    d["study_year"] = m.group(1) if m else None
    d["co2_only"] = bool(find_ef.data_warning(row))
    return d


def all_sources():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """SELECT s.source_id, s.name, s.publisher, s.version, s.year, s.license_type, s.url,
                      (SELECT COUNT(*) FROM emission_factors e WHERE e.source_id = s.source_id) AS n
               FROM sources s ORDER BY n DESC""").fetchall()
    finally:
        conn.close()
    return [dict(r) for r in rows]


def finalize(run_steps, matches, chosen):
    """Do the arithmetic. chosen: {step index: ef_id of a stand-in the person picked}.
    Returns (results, summary)."""
    calc_steps, problems, user_proxies = [], {}, set()
    for i, (s, m) in enumerate(zip(run_steps, matches)):
        ef_id, conf = m["ef_id"], m["confidence"]
        if m["problem"]:
            problems[i] = m["problem"]
        elif ef_id is None and i in chosen:
            ef_id, conf = int(chosen[i]), "low"
            user_proxies.add(i)
        item = {"step": s["label"], "quantity": s["quantity"], "unit": s["unit"],
                "ef_id": ef_id, "confidence": conf}
        if "distance_km" in s:
            item["distance_km"] = s["distance_km"]
        calc_steps.append(item)

    results, _ = calculate_footprint(calc_steps)
    for idx, why in problems.items():
        results[idx]["status"] = "cannot_calculate"
        results[idx]["reason"] = scrub(why, 400)
    for idx in user_proxies:
        results[idx]["user_proxy"] = True
        results[idx]["notes"].append("USER-CHOSEN PROXY: you picked this stand-in because the database "
                                     "has no exact factor for this step.")
    for i, r in enumerate(results):
        r["ai_reasoning"] = matches[i].get("reasoning", "")
        r["distance_note"] = run_steps[i].get("distance_note")
        r["trip_km"] = run_steps[i].get("trip_km")
        r["detail"] = factor_details(r["ef_id"]) if r.get("activity") else None
        r["amount_text"] = amount_text(run_steps[i])
    return results, summarise(results)


# ---------------------------------------------------------------------------
# Presentation helpers (plain text, used by the screen and the downloads)
# ---------------------------------------------------------------------------
def amount_text(step):
    if step.get("unit_kind") == "distance":
        return f"{num(step['quantity'])} km trip (no weight)"
    if step.get("unit_kind") == "tonne.km":
        return f"{num(step['quantity'])} {step['unit']} x {num(step['distance_km'])} km"
    return f"{num(step['quantity'])} {step['unit']}"


def status_text(r):
    if r["status"] == "clean":
        return f"{r['confidence']} confidence"
    if r["status"] == "proxy":
        if r.get("user_proxy"):
            return "your stand-in"
        if r.get("proxy_reason") == "co2_only":
            return "CO2-only factor"
        return "low confidence"
    if r["status"] == "no_match":
        return "no factor"
    return "could not calculate"


def source_label(d):
    """Short 'who published it, which year' text for the results table."""
    if not d:
        return "-"
    who = d.get("publisher") or d.get("source_name") or "unknown source"
    year = d.get("source_year") or ""
    if not year and d.get("study_year"):
        year = f"study {d['study_year']}"
    return f"{who}, {year}" if year else who


def source_markdown(entry):
    """Full citation lines for one source (used in the 'Sources' section)."""
    lines = [entry["name"]]
    if entry.get("publisher"):
        lines.append(f"Publisher: {entry['publisher']}")
    ver = " / ".join(str(x) for x in (entry.get("version"), entry.get("year")) if x)
    lines.append(f"Version / year: {ver}" if ver else "Version / year: not recorded in the database")
    if entry.get("license"):
        lines.append(f"Licence: {entry['license']}")
    lines.append(f"Link: {entry['url']}" if entry.get("url") else "Link: not recorded in the database")
    if entry.get("study_years"):
        lines.append("Underlying study year(s): " + ", ".join(sorted(entry["study_years"])))
    lines.append("Used in step(s): " + ", ".join(str(n) for n in entry["steps"]))
    return lines


def sources_used(results):
    """One entry per source actually used in this result, listing which steps used it."""
    seen = {}
    for i, r in enumerate(results, 1):
        d = r.get("detail")
        if not d:
            continue
        e = seen.setdefault(d["source_id"], {
            "name": d.get("source_name") or "unknown source", "publisher": d.get("publisher"),
            "version": d.get("source_version"), "year": d.get("source_year"),
            "license": d.get("license_type"), "url": d.get("source_url"),
            "steps": [], "study_years": set()})
        e["steps"].append(i)
        if d.get("study_year"):
            e["study_years"].add(d["study_year"])
    return list(seen.values())


def table_rows(results):
    rows = []
    for i, r in enumerate(results, 1):
        d = r.get("detail") or {}
        rows.append({
            "Step": i, "What": r["step"], "Amount": r["amount_text"],
            "Factor used": (f"{r['activity']} ({d.get('geography')})" if r.get("activity") else "none found"),
            "Source": source_label(d), "Confidence": status_text(r),
            "kg CO2e": r["contribution"],
        })
    return rows


# ---------------------------------------------------------------------------
# Downloads
# ---------------------------------------------------------------------------
def _cell(value):
    """Stop spreadsheet programs treating text as a formula."""
    text = "" if value is None else str(value)
    return "'" + text if text[:1] in ("=", "+", "-", "@") else text


def to_csv(question, results, summary):
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["Question", _cell(question)])
    w.writerow([])
    w.writerow(["Step", "What", "Amount", "Factor used", "Geography", "Factor value", "Factor unit",
                "Source", "Publisher", "Version", "Year", "Licence", "Link", "Database row",
                "Confidence", "kg CO2e", "Notes"])
    for i, r in enumerate(results, 1):
        d = r.get("detail") or {}
        notes = "; ".join(str(n) for n in r["notes"]) or (r.get("reason") or "")
        w.writerow([i, _cell(r["step"]), _cell(r["amount_text"]), _cell(r.get("activity")),
                    _cell(d.get("geography")), r.get("factor_value"), _cell(r.get("factor_unit")),
                    _cell(d.get("source_name")), _cell(d.get("publisher")), _cell(d.get("source_version")),
                    _cell(d.get("source_year") or d.get("study_year")), _cell(d.get("license_type")),
                    _cell(d.get("source_url")), d.get("ef_id", ""), status_text(r),
                    "" if r["contribution"] is None else f"{r['contribution']:.6f}", _cell(notes)])
    w.writerow([])
    covered = summary["step_count"] - len(summary["gaps"])
    if covered == 0:
        w.writerow(["TOTAL", "none - no step could be calculated (this is not zero)"])
    else:
        w.writerow(["Total, high/medium-confidence steps only (kg CO2e)", f"{summary['clean_total']:.6f}"])
        w.writerow(["Total including stand-ins and low-confidence (kg CO2e)", f"{summary['grand_total']:.6f}"])
        w.writerow(["Steps covered", f"{covered} of {summary['step_count']}"])
    return buf.getvalue()


def to_html(question, results, summary):
    """A self-contained report page (open it, then print or save as PDF from the browser)."""
    e = html.escape
    covered = summary["step_count"] - len(summary["gaps"])
    out = ["<!DOCTYPE html><html><head><meta charset='utf-8'><title>Carbon footprint result</title>",
           "<style>body{font-family:Arial,Helvetica,sans-serif;max-width:960px;margin:2em auto;padding:0 1em;color:#222}"
           "table{border-collapse:collapse;width:100%;margin:1em 0}th,td{border:1px solid #ccc;padding:6px 8px;"
           "text-align:left;vertical-align:top;font-size:14px}th{background:#f2f2f2}.big{font-size:22px;font-weight:bold}"
           ".warn{background:#fff4e5;border:1px solid #f0b960;padding:8px 12px;margin:1em 0}"
           ".small{color:#555;font-size:13px}</style></head><body>",
           "<h1>Carbon footprint result</h1>",
           f"<p><b>Question:</b> {e(question)}</p>",
           f"<p class='small'>Generated {datetime.date.today().isoformat()}. Screening estimate, not an audited figure.</p>"]
    if covered == 0:
        out.append("<div class='warn'><b>No total.</b> No step could be calculated. This is not zero.</div>")
    else:
        out.append(f"<p class='big'>{summary['clean_total']:,.4f} kg CO2e "
                   "<span class='small'>(high/medium-confidence steps only)</span></p>")
        out.append(f"<p class='big'>{summary['grand_total']:,.4f} kg CO2e "
                   "<span class='small'>(including stand-ins and low-confidence steps)</span></p>")
        out.append(f"<p class='small'>Covers {covered} of {summary['step_count']} steps.</p>")
    if summary["gaps"]:
        out.append(f"<div class='warn'>{len(summary['gaps'])} step(s) have no emission factor and are NOT "
                   "included (missing, not zero), so the totals understate the footprint.</div>")
    out.append("<h2>Steps</h2><table><tr><th>#</th><th>What</th><th>Amount</th><th>Factor used</th>"
               "<th>Source</th><th>Confidence</th><th>kg CO2e</th></tr>")
    for i, r in enumerate(results, 1):
        d = r.get("detail") or {}
        val = "-" if r["contribution"] is None else f"{r['contribution']:,.4f}"
        factor = (f"{e(str(r['activity']))} ({e(str(d.get('geography')))})<br>"
                  f"<span class='small'>{e(num(r['factor_value']))} {e(str(r['factor_unit']))}</span>"
                  if r.get("activity") else "none found")
        out.append(f"<tr><td>{i}</td><td>{e(r['step'])}</td><td>{e(r['amount_text'])}</td><td>{factor}</td>"
                   f"<td>{e(source_label(d))}</td><td>{e(status_text(r))}</td><td>{val}</td></tr>")
    out.append("</table><h2>Sources used</h2>")
    for s in sources_used(results):
        out.append("<p>" + "<br>".join(e(line) for line in source_markdown(s)) + "</p>")
    assumptions = [f"Distance for step {i}: {num(r['trip_km'])} km ({r['distance_note']})"
                   for i, r in enumerate(results, 1) if r.get("distance_note")]
    notes = [f"Step {i}: {n}" for i, r in enumerate(results, 1) for n in r["notes"]]
    if assumptions or notes:
        out.append("<h2>Assumptions and notes</h2><ul>")
        out.extend(f"<li>{e(a)}</li>" for a in assumptions + notes)
        out.append("</ul>")
    gaps = [f"Step {i}: {r['step']} - {r['reason']}" for i, r in enumerate(results, 1)
            if r["status"] in ("no_match", "cannot_calculate")]
    if gaps:
        out.append("<h2>Open gaps</h2><ul>" + "".join(f"<li>{e(g)}</li>" for g in gaps) + "</ul>")
    out.append("</body></html>")
    return "".join(out)