"""
ai_match.py - AI-assisted choice of the right emission-factor row (free Gemini version).

The AI only ever SELECTS an ef_id from the shortlist (or says none fit).
It never sees or handles emission factor values, and does no arithmetic.

Needs:  pip install google-genai
        a free API key from aistudio.google.com stored in the GEMINI_API_KEY environment variable
"""

import json
import os
import sys
import time

from google import genai
from google.genai import types

# "gemini-flash-latest" always points to Google's current Flash model.
# If it ever gives a "model not found" error, open aistudio.google.com, pick a Flash
# model there, and paste its exact name here.
MODEL = "gemini-flash-latest"
# Lighter model used automatically if the main one is overloaded.
BACKUP_MODEL = "gemini-flash-lite-latest"

# Only these columns are shown to the AI. Numbers (co2e_total etc.) are deliberately left out.
FIELDS_SHOWN = ["ef_id", "activity", "category", "variant", "unit",
                "geography", "lca_stage", "notes"]

SYSTEM_PROMPT = """You are helping build a supply-chain carbon footprint calculator.
You are given a process step and a shortlist of candidate emission-factor rows found
by a text search. Decide which single candidate is the correct match for the step,
or decide that none of them is.

Rules:
- Choose a candidate only if it describes the same activity as the step. Similar
  words are not enough (e.g. cotton cultivation is not cotton fabric production).
- Check that the unit makes sense for the step. Freight that carries a weight needs a
  per tonne.km factor, not vehicle.km or passenger.km. But if the step note says it is about
  trip distance only (no weight), a factor per vehicle-km (per km or per mile of vehicle
  travel) is the right unit and per tonne.km is not.
- If an LCA stage is given, prefer candidates in that stage; a mismatch is a warning sign.
- Geography: prefer an exact match. A row whose geography is global, world or unspecified
  is a neutral stand-in: it is NOT a geography mismatch, so never lower confidence for it.
  Only if the row is from a specific country that differs from the country named in the step
  or the target geography, confidence can be at most medium, and should be low or null when
  the activity depends strongly on location (electricity, agriculture, manufacturing).
  Mention any real geography mismatch in the reasoning.
- If the step does not say which size, type or variant it means, prefer a row labelled
  "Average" or the most generic row. Never pick one specific variant (a particular ship
  size, load factor or technology) just because it is first; if only specific variants are
  offered, use low confidence.
- If a row adds a detail that the step does not mention and that changes the process
  (e.g. "coloured" yarn for a plain spinning step), that is a compromise: at most low confidence.
- Some rows already include upstream stages (e.g. a yarn row whose variant mentions cotton
  cultivation). Such a row only fits a step that covers that whole chain; for a step that
  covers just one stage it would double count, so use low confidence or null.
- If the step states a specific qualifier (for example electric, refrigerated, diesel,
  organic, container ship) or a vehicle class (truck vs van), a row that contradicts it is a
  different activity, not a compromise. Do not use a diesel vehicle for an electric one, or a
  van for a truck, even at low confidence: return null.
- If no candidate is a good fit, return null. Never force a pick. Null is a correct answer.
- Confidence: high = clearly the right row; medium = right activity but some
  compromise (geography, variant); low = a weak or uncertain fit.
- You only select an ef_id. Never invent an ef_id and never mention or compute
  emission values.
- Keep the reasoning to 1-3 short sentences."""

# The "form" the AI must fill in. The API enforces this shape.
DECISION_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "match": {"type": "integer", "nullable": True},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
    },
    "required": ["reasoning", "match", "confidence"],
}


def _trim(candidate):
    """Keep only the descriptive fields (no numbers) and shorten long notes."""
    slim = {k: candidate.get(k) for k in FIELDS_SHOWN}
    if slim["notes"] and len(slim["notes"]) > 300:
        slim["notes"] = slim["notes"][:300] + "..."
    return slim


def choose_match(step, candidates, stage=None, model=MODEL, geography=None):
    """
    step       : text description of the process step, e.g. "raw cotton cultivation"
    candidates : list of dicts (rows from find_ef), each with at least an 'ef_id'
    stage      : LCA stage if known (e.g. "raw_material_extraction"), else None

    Returns a dict: {"match": ef_id or None, "confidence": ..., "reasoning": ...}
    """
    if not candidates:
        return {"match": None, "confidence": "low",
                "reasoning": "No candidates were supplied, so no AI call was made."}

    valid_ids = {int(c["ef_id"]) for c in candidates}
    shown = [_trim(c) for c in candidates]

    user_text = (
        f"Process step: {step}\n"
        f"LCA stage: {stage or 'not known'}\n"
        f"Target geography of the supply chain: {geography or 'not specified'}\n\n"
        f"Candidate rows (JSON):\n{json.dumps(shown, indent=2)}"
    )

    client = genai.Client()  # reads GEMINI_API_KEY automatically
    decision = None
    last_error = None
    # Try the main model up to 3 times, then the lighter backup model up to 3 times.
    for model_name in (model, BACKUP_MODEL):
        for attempt in range(3):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=user_text,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT,
                        temperature=0,  # same input should give the same answer
                        # These two lines FORCE the reply to follow the form above.
                        response_mime_type="application/json",
                        response_schema=DECISION_SCHEMA,
                    ),
                )
                decision = json.loads(response.text)
                break
            except Exception as e:
                last_error = e
                busy = any(word in str(e) for word in ("503", "429", "UNAVAILABLE"))
                if not busy:
                    break  # a different problem (e.g. bad key): retrying won't help
                time.sleep(5 * (attempt + 1))  # wait 5s, then 10s, then 15s
        if decision is not None:
            break

    if decision is None:
        return {"match": None, "confidence": "low",
                "reasoning": f"AI call failed after retries: {last_error}", "error": True}

    if not isinstance(decision, dict):
        return {"match": None, "confidence": "low",
                "reasoning": "Model did not return a structured decision.", "error": True}

    match = decision.get("match")
    confidence = decision.get("confidence")
    reasoning = str(decision.get("reasoning", ""))

    # Safety check: the pick must be one of the candidates we actually sent.
    if match is not None:
        try:
            match = int(match)
        except (TypeError, ValueError):
            match = None
        if match not in valid_ids:
            reasoning += " [Rejected: model returned an ef_id that was not in the shortlist.]"
            match = None
    if confidence not in ("high", "medium", "low"):
        confidence = "low"

    return {"match": match, "confidence": confidence, "reasoning": reasoning}


QUERY_PROMPT = """You help search a database of emission factors. A text search for a process
step did not find a suitable row. Suggest up to 3 different short search phrases (1-4 words
each) that could find the right row IF it exists, using wording a database would use
(for example: HGV for truck, clothing for garment, fiber for fibre, grid electricity for
power). Use words different from the original step. You only suggest search phrases:
never invent emission values."""

QUERY_SCHEMA = {
    "type": "object",
    "properties": {"queries": {"type": "array", "items": {"type": "string"}}},
    "required": ["queries"],
}


def suggest_queries(step, stage=None, model=MODEL):
    """Ask the AI for alternative search phrases. Returns a list of up to 3 strings
    (empty list if the AI call fails). The AI never sees the database here."""
    client = genai.Client()
    text = f"Process step: {step}\nLCA stage: {stage or 'not known'}"
    for model_name in (model, BACKUP_MODEL):
        for attempt in range(3):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=text,
                    config=types.GenerateContentConfig(
                        system_instruction=QUERY_PROMPT,
                        temperature=0,
                        response_mime_type="application/json",
                        response_schema=QUERY_SCHEMA,
                    ),
                )
                queries = json.loads(response.text).get("queries", [])
                return [str(q).strip() for q in queries if str(q).strip()][:3]
            except Exception as e:
                busy = any(word in str(e) for word in ("503", "429", "UNAVAILABLE"))
                if not busy:
                    return []
                time.sleep(5 * (attempt + 1))
    return []