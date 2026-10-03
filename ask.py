"""
ask.py - ask the carbon calculator a question in plain English.

Examples:
    python ask.py "Give me the emission to transport 100 kg cotton from Bangalore to Chennai on an EV truck"
    python ask.py "emissions of 250 kWh of grid electricity in India"
or just run   python ask.py   and type your question when asked.

HOW IT WORKS
  1. The AI READS your question and turns it into structured steps (what, how much, how far).
     It never does arithmetic. Every number it reports must appear in your question,
     otherwise the script stops and tells you.
  2. If a freight step needs a distance and you did not give one, the AI can offer a rough
     estimate from the place names - but it is shown as UNVERIFIED and you must accept or
     replace it. (Better still: put the distance in your question, e.g. "... 350 km ...".)
  3. Each step then goes through the same pipeline as run_pipeline.py: search -> AI picks
     the database row (or says none fit) -> plain arithmetic by calculate_footprint.py.
  4. If the database has no exact factor, the AI suggests the closest stand-ins it does have
     and explains how each differs. YOU choose one (or none). A stand-in you choose is
     labelled USER-CHOSEN PROXY and counts in the grand total only. Pressing Enter leaves the
     step as a gap. Nothing is ever picked for you, and nothing is ever counted as zero.

Needs the same folder as find_ef.py, ai_match.py, run_pipeline.py, calculate_footprint.py and
emission_factors.db, and GEMINI_API_KEY set.

Options:  --yes   accept the AI's distance estimate without asking (it is still flagged).
                  Stand-ins are then shown for information only, never used.
          --geo   where the activity happens if your question does not say (default: India)
"""

import argparse
import json
import os
import re
import sys
import time

from google import genai
from google.genai import types

import ai_match
import calculate_footprint as cfmod
import run_pipeline
from ai_match import MODEL, BACKUP_MODEL, _trim
from calculate_footprint import (calculate_footprint, print_report, summarise, num,
                                 factor_denominator, convert_quantity, CalcError)

# ask.py relies on two rules that live in ai_match.py's instructions to the AI. If ai_match.py is
# an older version, stop with a clear message instead of quietly giving weaker answers.
REQUIRED_RULES = ("different activity, not a compromise", "neutral stand-in", "trip distance only")

DEFAULT_GEOGRAPHY = run_pipeline.TARGET_GEOGRAPHY
STAGES = ["raw_material_extraction", "intermediate_processing", "manufacturing_assembly",
          "transport", "end_of_life"]

PARSE_PROMPT = """You read a person's question about the carbon footprint of one or more
supply-chain activities and turn it into structured steps. You only READ the question.
Copy every number and unit exactly as written. Never convert units, never calculate, and
never invent a number that is not in the question (the one exception is
distance_km_estimate, described below).

Return one step per distinct activity, with:
- search_text: 2 to 6 words naming the activity the way an emission-factor database would,
  keeping every specific the person stated (e.g. "electric truck freight",
  "refrigerated HGV freight", "grid electricity", "cotton fibre production").
- stage: one of raw_material_extraction, intermediate_processing, manufacturing_assembly,
  transport, end_of_life; or null if unsure. Moving goods = transport.
- quantity: the amount exactly as written (a number), or null if not stated. For freight
  this is the weight of the goods being moved, or null if no weight is given.
- unit: the unit exactly as written for that quantity (kg, tonne, kWh, ...), or null.
- is_freight: true if the step moves goods over a distance.
- freight_basis: for every freight step, "mass_distance" if a weight of goods is given (alone
  or together with a material), otherwise "distance_only". null for steps that are not freight.
- distance_stated: the distance number if the question states one, else null.
- distance_unit: "km" or "miles", as written, if a distance is stated.
- route_from, route_to: the place names if the question gives a route, else null.
- distance_km_estimate: ONLY if is_freight is true, no distance is stated, and both places
  are named: your best estimate of the typical road distance in km between them (a whole
  number). Otherwise null. The person will be told it is an unverified estimate.
- geography: the country or region where the activity takes place if it is clear
  (e.g. "India"), else null.
Materials and transport:
- If the person names a material or product AND gives its weight AND asks for transport,
  return TWO steps, in this order: (1) the material itself (its production or extraction,
  stage raw_material_extraction unless clearly something else) with that weight, then (2) the
  freight step carrying the same weight. Example: "transport 200 kg of cotton from A to B by
  truck" gives a cotton fibre production step (200 kg) and a truck freight step (200 kg).
- If a weight is given but NO material is named, return only the freight step with that weight.
- If neither a material nor a weight is given (for example "transport something from A to B"),
  return only the freight step, with quantity null, unit null and freight_basis
  "distance_only". If no vehicle or mode is named, assume road and use "truck freight".
If the question is too unclear to turn into steps, return no steps and put a short
question for the person in "clarification"."""

_NUM = {"type": "number", "nullable": True}
_STR = {"type": "string", "nullable": True}
PARSE_SCHEMA = {
    "type": "object",
    "properties": {
        "steps": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "search_text": {"type": "string"},
                "stage": _STR, "quantity": _NUM, "unit": _STR,
                "is_freight": {"type": "boolean"}, "freight_basis": _STR,
                "distance_stated": _NUM, "distance_unit": _STR,
                "route_from": _STR, "route_to": _STR,
                "distance_km_estimate": _NUM, "geography": _STR,
            },
            "required": ["search_text", "is_freight"],
        }},
        "clarification": _STR,
    },
    "required": ["steps"],
}

WIDE_TOP_N = 40      # rows per single-word search when looking for stand-ins

PROXY_PROMPT = """You are helping a person choose a STAND-IN emission factor because the database
has no exact match for their process step. You get the step and a list of candidate rows (all
already in a unit that fits the step).

Pick up to 3 candidates that a careful life-cycle-assessment practitioner might reasonably
use as a stand-in, best first. For each give:
- why_close: what it has in common with the step (one short sentence).
- how_it_differs: the most important difference from what the person asked for (one short
  sentence), e.g. different fuel or powertrain, vehicle class, country, or process boundary.
- likely_bias: "overstates", "understates" or "unclear": whether using this row probably gives
  a higher or lower footprint than the true activity. Judge only from differences you can see,
  never from numbers. Vehicle size matters: per tonne.km, small vehicles (vans, small rigid
  trucks) usually emit MORE than large trucks, and large vehicles usually LESS. If the step
  does not name a size, use the size shown in each row to say whether it likely overstates
  (much smaller than a typical freight truck) or understates (much larger). If the direction
  depends on information you cannot see (for example how clean the electricity grid is),
  say "unclear".
If no candidate is a reasonable stand-in, return an empty list. An empty list is a good answer.
Never invent an ef_id, and never mention or compute emission values. Keep sentences short."""

PROXY_SCHEMA = {
    "type": "object",
    "properties": {"options": {"type": "array", "items": {
        "type": "object",
        "properties": {
            "ef_id": {"type": "integer"},
            "why_close": {"type": "string"},
            "how_it_differs": {"type": "string"},
            "likely_bias": {"type": "string", "enum": ["overstates", "understates", "unclear"]},
        },
        "required": ["ef_id", "why_close", "how_it_differs", "likely_bias"],
    }}},
    "required": ["options"],
}


class Cancelled(Exception):
    pass


def ask(prompt):
    """input() that returns None if there is no keyboard to type on."""
    try:
        return input(prompt).strip()
    except EOFError:
        return None


def numbers_in(text):
    return [float(x.replace(",", "")) for x in re.findall(r"\d[\d,]*(?:\.\d+)?", text)]


def appears_in(value, numbers):
    return any(abs(float(value) - n) < 1e-9 for n in numbers)


def parse_question(question):
    """Ask the AI to read the question. Returns (data, None) or (None, error text)."""
    client = genai.Client()
    run_pipeline.throttle()
    last_error = None
    for model_name in (MODEL, BACKUP_MODEL):
        for attempt in range(3):
            try:
                response = client.models.generate_content(
                    model=model_name, contents=question,
                    config=types.GenerateContentConfig(
                        system_instruction=PARSE_PROMPT, temperature=0,
                        response_mime_type="application/json", response_schema=PARSE_SCHEMA))
                data = json.loads(response.text)
                if not isinstance(data, dict):
                    raise ValueError("the AI did not return a structured reading")
                return data, None
            except Exception as e:
                last_error = e
                if not any(w in str(e) for w in ("503", "429", "UNAVAILABLE")):
                    break
                time.sleep(5 * (attempt + 1))
    return None, str(last_error)


def resolve_distance(p, qnums, auto_yes):
    """Work out the distance in km for a freight step.
    Returns (km, how_we_got_it). Raises Cancelled, or ValueError with a plain message."""
    stated = p.get("distance_stated")
    if stated is not None:
        if not appears_in(stated, qnums):
            raise ValueError(f"the AI read a distance of {num(stated)} but that number is not "
                             "in your question - please write the distance in digits")
        unit = str(p.get("distance_unit") or "km").strip().lower()
        if unit in ("km", "kms", "kilometre", "kilometres", "kilometer", "kilometers"):
            return float(stated), "stated in your question"
        if unit in ("mile", "miles", "mi"):
            return float(stated) * 1.609344, f"stated in your question ({num(stated)} miles, converted to km)"
        raise ValueError(f"distance unit '{unit}' not understood - use km or miles")

    frm, to, est = p.get("route_from"), p.get("route_to"), p.get("distance_km_estimate")
    route = f"{frm} -> {to}" if frm and to else "this journey"
    has_estimate = est is not None and est > 0 and frm and to

    if auto_yes:
        if has_estimate:
            return float(est), "AI ESTIMATE, UNVERIFIED (accepted automatically with --yes)"
        raise ValueError("no distance in your question and no route to estimate from - "
                         "add the distance, e.g. '350 km'")

    for _ in range(3):
        if has_estimate:
            reply = ask(f"   Distance {route}: about {num(est)} km by road (AI estimate, UNVERIFIED).\n"
                        "   Press Enter to accept, type your own distance in km, or n to cancel: ")
        else:
            reply = ask(f"   How far is the journey ({route}) in km? (n to cancel): ")
        if reply is None:
            raise ValueError("cannot ask for the distance here - add it to your question "
                             "(e.g. '350 km') or use --yes")
        if reply.lower() in ("n", "no", "cancel", "q"):
            raise Cancelled()
        if reply == "" and has_estimate:
            return float(est), "AI ESTIMATE, UNVERIFIED (accepted by you)"
        try:
            km = float(reply.replace(",", ""))
            if km > 0:
                return km, "entered by you"
        except ValueError:
            pass
        print("   Please type a positive number of km.")
    raise ValueError("no usable distance was given")


def read_steps(question, auto_yes, default_geo):
    """Turn the question into a list of ready-to-run steps. Returns (steps, problem_text)."""
    data, err = parse_question(question)
    if err:
        return None, f"the AI could not read your question ({err}). Try again in a moment."
    parsed = data.get("steps") or []
    if not parsed:
        return None, data.get("clarification") or "I could not tell what to calculate. Could you rephrase?"

    qnums = numbers_in(question)
    steps = []
    print("HOW I READ YOUR QUESTION")
    print("-" * 78)
    for n, p in enumerate(parsed, 1):
        label = str(p.get("search_text") or "").strip()
        if not label:
            return None, "I could not work out what activity you mean. Could you rephrase?"
        q, u = p.get("quantity"), p.get("unit")
        is_freight = bool(p.get("is_freight"))
        basis = p.get("freight_basis")
        if is_freight and basis not in ("mass_distance", "distance_only"):
            basis = "mass_distance" if (q is not None and u) else "distance_only"
        distance_only = is_freight and basis == "distance_only"

        stage = p.get("stage") if p.get("stage") in STAGES else None
        step = {"label": label, "geography": p.get("geography") or default_geo,
                "distance_note": None, "unit_kind": None, "unit_note": None}
        frm, to = p.get("route_from"), p.get("route_to")
        route = f"{frm} -> {to}, " if frm and to else ""
        print(f"  Step {n}: {label}")

        if distance_only:
            # Trip only, no weight: the quantity IS the distance, and the factor is per km of vehicle travel.
            try:
                km, how = resolve_distance(p, qnums, auto_yes)
            except ValueError as e:
                return None, f"step {n} ({label}): {e}"
            stage = "transport"
            step.update(quantity=float(km), unit="km", trip_km=km, distance_note=how,
                        unit_kind="distance",
                        unit_note=("trip distance only, no weight: the factor must be per vehicle-km "
                                   "(per km or per mile of vehicle travel), not per tonne.km"))
            print(f"     Basis: trip distance only (per vehicle-km factors; no weight is used)"
                  f"   |   geography: {step['geography']}")
            print(f"     Distance: {route}{num(km)} km  [{how}]")
        else:
            if q is None or not u:
                return None, (f"step {n} ({label}): I could not find a quantity and unit. "
                              "Please include one, e.g. '100 kg' or '250 kWh'.")
            if not appears_in(q, qnums):
                return None, (f"step {n} ({label}): the AI read the quantity as {num(q)}, but that number "
                              "is not in your question. Please write the quantity in digits.")
            step.update(quantity=float(q), unit=str(u).strip())
            print(f"     Quantity: {num(q)} {step['unit']}   |   geography: {step['geography']}")
            if is_freight:
                try:
                    km, how = resolve_distance(p, qnums, auto_yes)
                except ValueError as e:
                    return None, f"step {n} ({label}): {e}"
                stage = "transport"
                step.update(distance_km=km, trip_km=km, distance_note=how, unit_kind="tonne.km",
                            unit_note="freight carrying a weight: the factor must be per tonne.km")
                print("     Basis: weight x distance (per tonne.km factors)")
                print(f"     Distance: {route}{num(km)} km  [{how}]")
        step["stage"] = stage
        steps.append(step)
    print()
    return steps, None


def widen_pool(step, pool):
    """Add rows found by searching each word of the step on its own (with a bigger cap), so the
    closest stand-ins are not missed just because earlier searches ranked other rows first.
    Example: for 'electric truck freight', the word 'electric' alone finds the electric vans."""
    have = {int(r["ef_id"]) for r in pool}
    wide = list(pool)
    for word in step["label"].split():
        rows, _ = run_pipeline.search(word, step["stage"], WIDE_TOP_N, step.get("unit_kind"))
        for r in rows:
            if int(r["ef_id"]) not in have:
                have.add(int(r["ef_id"]))
                wide.append(r)
    return wide


def usable_rows(pool, step):
    """Rows the search saw whose unit fits this step, so the calculator can really use them
    (e.g. a per-tonne.km freight row for a freight step; per-km rows are dropped)."""
    rows, seen = [], set()
    for r in pool:
        eid = int(r["ef_id"])
        if eid in seen or r.get("co2e_total") is None:
            continue
        seen.add(eid)
        try:
            convert_quantity(step["quantity"], step["unit"], factor_denominator(r["co2e_unit"]),
                             step.get("distance_km"))
        except CalcError:
            continue
        rows.append(r)
    return rows[:40]


def suggest_proxy_options(step, rows):
    """Ask the AI which of these rows are the most reasonable stand-ins and how each differs.
    The AI sees no numbers. Returns (options, None) or (None, error text)."""
    client = genai.Client()
    run_pipeline.throttle()
    text = (f"Process step: {step['label']}\nLCA stage: {step['stage'] or 'not known'}\n"
            f"Target geography: {step['geography']}\n\n"
            f"Candidate rows (JSON):\n{json.dumps([_trim(r) for r in rows], indent=2)}")
    last_error = None
    data = None
    for model_name in (MODEL, BACKUP_MODEL):
        for attempt in range(3):
            try:
                response = client.models.generate_content(
                    model=model_name, contents=text,
                    config=types.GenerateContentConfig(
                        system_instruction=PROXY_PROMPT, temperature=0,
                        response_mime_type="application/json", response_schema=PROXY_SCHEMA))
                data = json.loads(response.text)
                break
            except Exception as e:
                last_error = e
                if not any(w in str(e) for w in ("503", "429", "UNAVAILABLE")):
                    break
                time.sleep(5 * (attempt + 1))
        if data is not None:
            break
    if not isinstance(data, dict):
        return None, str(last_error)
    valid = {int(r["ef_id"]) for r in rows}
    options = []
    for o in data.get("options") or []:
        try:
            eid = int(o.get("ef_id"))
        except (TypeError, ValueError):
            continue
        if eid in valid and all(eid != x["ef_id"] for x in options):
            bias = o.get("likely_bias") if o.get("likely_bias") in ("overstates", "understates") else "unclear"
            options.append({"ef_id": eid, "why_close": str(o.get("why_close", "")),
                            "how_it_differs": str(o.get("how_it_differs", "")), "likely_bias": bias})
    return options[:3], None


def short(text, n=70):
    text = "" if text is None else str(text)
    return text if len(text) <= n else text[: n - 3] + "..."


def offer_proxies(step, pool, allow_choice):
    """No exact match: show the closest stand-ins and let the person choose one.
    Returns the chosen database row, or None (the step stays an open gap)."""
    print("   STAND-IN OPTIONS - the database has no exact factor for this step.")
    rows = usable_rows(widen_pool(step, pool), step)
    if not rows:
        print("   None of the rows the search found are in a unit that fits this step.")
        print("   It stays an open gap.")
        return None
    options, err = suggest_proxy_options(step, rows)
    if err:
        print(f"   Could not get stand-in suggestions ({err}). It stays an open gap.")
        return None
    if not options:
        print("   The AI sees no reasonable stand-in among the rows found. It stays an open gap.")
        return None

    by_id = {int(r["ef_id"]): r for r in rows}
    print("   Closest options in your database (the AI explains them; the numbers come from the database):")
    for n, o in enumerate(options, 1):
        r = by_id[o["ef_id"]]
        converted, _ = convert_quantity(step["quantity"], step["unit"],
                                        factor_denominator(r["co2e_unit"]), step.get("distance_km"))
        name = " | ".join(short(x, 45) for x in (r["activity"], r["category"], r["variant"]) if x)
        print(f"   [{n}] {name}")
        print(f"       ef_id {r['ef_id']}, {r['geography']}; factor {num(r['co2e_total'])} {r['co2e_unit']}"
              f"  ->  would give {converted * r['co2e_total']:.4f} kg CO2e for this step")
        print(f"       Close because: {o['why_close']}")
        print(f"       Differs: {o['how_it_differs']}")
        print(f"       Likely effect (AI opinion, unverified): {o['likely_bias']}")

    if not allow_choice:
        print("   (Shown for information only - not used. Run without --yes to choose one.)")
        return None
    print("   Only pick one you could defend in a report; it will be labelled USER-CHOSEN PROXY.")
    for _ in range(3):
        if len(options) == 1:
            reply = ask("   Type 1 to use it, or press Enter to leave this step as a gap: ")
        else:
            reply = ask(f"   Choose 1-{len(options)}, or press Enter to leave this step as a gap: ")
        if reply is None:
            print("   No keyboard available - leaving this step as a gap.")
            return None
        if reply == "" or reply.lower() in ("n", "no", "none", "0"):
            print("   Left as an open gap.")
            return None
        if reply.isdigit() and 1 <= int(reply) <= len(options):
            return by_id[options[int(reply) - 1]["ef_id"]]
        print("   Please type one of the option numbers, or press Enter.")
    print("   No valid choice - leaving this step as a gap.")
    return None


def run_steps(steps, allow_choice=True):
    """Match every step (search + AI) and calculate. Returns (results, summary)."""
    print("=" * 78)
    print("PART 1 - MATCHING (search + AI)")
    print("=" * 78)
    calc_steps, problems, user_proxies = [], {}, set()
    for i, s in enumerate(steps, 1):
        print(f"STEP {i}: {s['label']}    (stage filter: {s['stage'] or 'none'}, "
              f"geography: {s['geography']})", flush=True)
        run_pipeline.TARGET_GEOGRAPHY = s["geography"]
        m = run_pipeline.match_step(s["label"], s["stage"], s.get("unit_kind"), s.get("unit_note"))
        for note in m["notes"]:
            print(f"   {note}")
        if m["problem"]:
            print(f"   PROBLEM: {m['problem']}")
        elif m["ef_id"] is None:
            print("   DECISION: NO MATCH - needs external source")
            print(f"   AI reasoning: {m['reasoning']}")
            chosen = offer_proxies(s, m.get("pool", []), allow_choice)
            if chosen is not None:
                m["ef_id"], m["confidence"], m["row"] = int(chosen["ef_id"]), "low", chosen
                user_proxies.add(i - 1)
                print(f"   DECISION (yours): USER-CHOSEN PROXY ef_id {m['ef_id']} - {chosen['activity']} "
                      f"({chosen['geography']})")
        else:
            row = m["row"]
            print(f"   DECISION: MATCHED ef_id {m['ef_id']} - {row['activity']} "
                  f"({row['geography']}), confidence: {m['confidence']}")
            print(f"   AI reasoning: {m['reasoning']}")
        print()
        item = {"step": s["label"], "quantity": s["quantity"], "unit": s["unit"],
                "ef_id": m["ef_id"], "confidence": m["confidence"]}
        if "distance_km" in s:
            item["distance_km"] = s["distance_km"]
        calc_steps.append(item)
        if m["problem"]:
            problems[i - 1] = m["problem"]

    print("PART 2 - CALCULATION (plain arithmetic, no AI)")
    results, _ = calculate_footprint(calc_steps)
    for idx, why in problems.items():
        results[idx]["status"] = "cannot_calculate"
        results[idx]["reason"] = why
    for idx in user_proxies:
        results[idx]["user_proxy"] = True
        results[idx]["notes"].append("USER-CHOSEN PROXY: you picked this stand-in because the database "
                                     "has no exact factor for this step.")
    summary = summarise(results)
    print_report(results, summary)
    return results, summary


def print_answer(question, steps, results, summary):
    print()
    print("=" * 78)
    print("ANSWER")
    print("=" * 78)
    print(f"  Your question: {question}")
    print()
    for n, (s, r) in enumerate(zip(steps, results), 1):
        print(f"  Step {n}: {s['label']}  ({num(s['quantity'])} {s['unit']})")
        if r["status"] == "clean":
            co2_note = "; CO2-only factor" if any(str(x).startswith("CO2 only") for x in r["notes"]) else ""
            print(f"          {r['contribution']:.4f} kg CO2e   ({r['confidence']} confidence; "
                  f"factor: {r['activity']}, {r['geography']}{co2_note})")
        elif r["status"] == "proxy" and r.get("proxy_reason") == "co2_only":
            print(f"          {r['contribution']:.4f} kg CO2e   CO2-ONLY FACTOR - no methane or nitrous "
                  f"oxide, so probably understates (factor: {r['activity']}, {r['geography']})")
        elif r["status"] == "proxy":
            kind = "USER-CHOSEN PROXY" if r.get("user_proxy") else "LOW-CONFIDENCE PROXY"
            print(f"          {r['contribution']:.4f} kg CO2e   {kind} - treat as a rough "
                  f"stand-in ({r['activity']}, {r['geography']})")
        else:
            print(f"          NO ANSWER: {r['reason']}")
            print("          Nothing was guessed and it is not counted as zero.")

    n_steps = summary["step_count"]
    covered = n_steps - len(summary["gaps"])
    if n_steps > 1 or covered == 0:
        print()
        print("  " + "-" * 74)
        if covered == 0:
            print("  TOTAL: none - no step could be calculated (this is NOT zero).")
        else:
            print(f"  Total, high/medium-confidence steps only  : {summary['clean_total']:.4f} kg CO2e"
                  f"   ({summary['clean_count']} of {n_steps} steps)")
            print(f"  Total incl. stand-ins and low-confidence  : {summary['grand_total']:.4f} kg CO2e"
                  f"   ({covered} of {n_steps} steps)")
            if summary["proxy_count"]:
                kinds = ("a stand-in, low-confidence or CO2-only factor" if cfmod.DEMOTE_CO2_ONLY
                         else "a stand-in or low-confidence factor")
                print(f"  ({summary['proxy_count']} step(s) rest on {kinds}: "
                      f"{summary['proxy_total']:.4f} kg CO2e of the second total.)")
            if summary["gaps"]:
                print(f"  WARNING: {len(summary['gaps'])} step(s) have no factor and are NOT included "
                      "(missing, not zero), so both totals understate the footprint.")

    notes = [(s["label"], s["trip_km"], s["distance_note"]) for s in steps if s.get("distance_note")]
    if notes:
        print()
        print("  ASSUMPTIONS TO CHECK:")
        for label, km, how in notes:
            print(f"  - distance for '{label}': {num(km)} km [{how}]")


def handle(question, args):
    print()
    try:
        steps, problem = read_steps(question, args.yes, args.geo)
    except Cancelled:
        print("Cancelled.")
        return
    if problem:
        print(f"Could not go ahead: {problem}")
        return
    results, summary = run_steps(steps, allow_choice=not args.yes)
    print_answer(question, steps, results, summary)


def main():
    sys.stdout.reconfigure(errors="replace")
    parser = argparse.ArgumentParser(description="Ask the carbon calculator a question in plain English.")
    parser.add_argument("question", nargs="*", help="your question, in quotes")
    parser.add_argument("--yes", action="store_true",
                        help="accept the AI's distance estimate without asking (still flagged)")
    parser.add_argument("--geo", default=DEFAULT_GEOGRAPHY,
                        help=f"where the activity happens if the question does not say (default {DEFAULT_GEOGRAPHY})")
    args = parser.parse_args()

    if not all(rule in ai_match.SYSTEM_PROMPT for rule in REQUIRED_RULES):
        sys.exit("ai_match.py is an older version. Replace it with the newest ai_match.py, "
                 "then run this again.")
    if not os.environ.get("GEMINI_API_KEY"):
        sys.exit("No API key found. Set the GEMINI_API_KEY environment variable first.")

    if args.question:
        handle(" ".join(args.question), args)
    else:
        while True:
            q = ask("\nAsk a question (or press Enter to quit): ")
            if not q:
                break
            handle(q, args)


if __name__ == "__main__":
    main()