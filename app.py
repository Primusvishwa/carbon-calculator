"""
app.py - web interface for the carbon footprint calculator (Streamlit).

Run on your own computer:   streamlit run app.py
Or deploy it free on Streamlit Community Cloud (put your Gemini key in the app's Secrets box as
    GEMINI_API_KEY = "your-key"
- never write the key into any file).

Optional settings (in Secrets, same format):
    MAX_QUESTIONS_PER_DAY     = 40     questions allowed per day, for everybody together
    MAX_QUESTIONS_PER_SESSION = 8      questions allowed per visitor session
    AI_MIN_GAP_SECONDS        = 6.0    pause between AI calls (free Gemini tier: about 10 a minute)
"""

import datetime
import os
import threading

import streamlit as st

st.set_page_config(page_title="Carbon footprint calculator", page_icon="🌿", layout="wide")

try:    # on Streamlit Cloud the key comes from the Secrets box; locally from an environment variable
    if "GEMINI_API_KEY" in st.secrets:
        os.environ["GEMINI_API_KEY"] = str(st.secrets["GEMINI_API_KEY"])
except Exception:
    pass


def setting(name, default):
    try:
        if name in st.secrets:
            return type(default)(st.secrets[name])
    except Exception:
        pass
    return type(default)(os.environ.get(name, default))


import pandas as pd  # noqa: E402  (comes with Streamlit)

import ai_match  # noqa: E402
import ask  # noqa: E402
import engine  # noqa: E402

engine.AI_MIN_GAP_SECONDS = setting("AI_MIN_GAP_SECONDS", 6.0)
SESSION_LIMIT = setting("MAX_QUESTIONS_PER_SESSION", 8)
DAILY_LIMIT = setting("MAX_QUESTIONS_PER_DAY", 40)
DEFAULT_GEO = "India"

EXAMPLES = [
    "How much emission to transport 200 kg of cotton from Chennai to Chengalpattu by truck?",
    "Emissions of 250 kWh of grid electricity in India",
    "How much emission to transport something from Chennai to Chengalpattu?",
]

# ---------------------------------------------------------------------------
# Start-up checks
# ---------------------------------------------------------------------------
if not all(rule in ai_match.SYSTEM_PROMPT for rule in ask.REQUIRED_RULES):
    st.error("This app's files are out of date (ai_match.py). The owner needs to update them.")
    st.stop()
if not os.environ.get("GEMINI_API_KEY"):
    st.error("This app has no Gemini key set up yet. The owner needs to add GEMINI_API_KEY in the app's Secrets.")
    st.stop()


@st.cache_resource
def _usage():
    return {"day": "", "count": 0, "lock": threading.Lock()}


def take_daily_slot():
    """Count one question against the shared daily limit. False if the limit is used up."""
    u = _usage()
    today = datetime.date.today().isoformat()
    with u["lock"]:
        if u["day"] != today:
            u["day"], u["count"] = today, 0
        if u["count"] >= DAILY_LIMIT:
            return False
        u["count"] += 1
        return True


def esc(text):
    """Stop database or user text from being read as formatting."""
    out = "" if text is None else str(text)
    for ch in ("\\", "*", "_", "`", "$", "[", "]", "#", "<", ">"):
        out = out.replace(ch, "\\" + ch)
    return out


def fmt_kg(x):
    return f"{x:,.2f}" if abs(x) >= 1 else f"{x:.4f}"


ss = st.session_state
ss.setdefault("stage", "ask")
ss.setdefault("asked", 0)
ss.setdefault("q_text", "")


def restart():
    for k in ("question", "parsed", "run_steps", "matches", "options", "chosen", "final"):
        ss.pop(k, None)
    ss.stage = "ask"


def use_example(text):
    ss.q_text = text


# ---------------------------------------------------------------------------
# Side bar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("About this tool")
    st.write("You ask a question in plain English. An AI reads it and picks the best-matching emission "
             "factors from a curated database. The arithmetic is plain code, and every number shows "
             "which source it came from.")
    st.write("If the database has no factor for something, you are told so. Nothing is guessed, and "
             "a missing factor is never counted as zero.")
    st.write("Questions are processed with Google's Gemini API, so please do not type anything private.")
    with st.expander("Data sources in the database"):
        for s in engine.all_sources():
            st.markdown(f"**{esc(s['name'])}**  \n{esc(s['publisher'])}  \n"
                        f"{s['n']} factors  \nLicence: {esc(s['license_type'])}"
                        + (f"  \n[{s['url']}]({s['url']})" if s.get("url") else ""))
    if ss.stage != "ask":
        st.button("Start over", key="restart_side", on_click=restart)

st.title("Carbon footprint calculator")
st.caption("Ask in plain English. Every factor is shown with its source. Screening estimates, not audited figures.")

# ---------------------------------------------------------------------------
# Stage: ask
# ---------------------------------------------------------------------------
if ss.stage == "ask":
    st.text_area("Your question", key="q_text", height=90, max_chars=engine.MAX_QUESTION_CHARS,
                 placeholder="e.g. How much emission to transport 200 kg of cotton from Chennai to "
                             "Chengalpattu by truck?")
    st.caption("Or try an example:")
    for n, ex in enumerate(EXAMPLES):
        st.button(ex, key=f"ex_{n}", on_click=use_example, args=(ex,))
    st.caption("Tip: put distances in the question (for example '350 km') for the most reliable result. "
               "If you leave them out, the AI offers an estimate that you can check and change.")

    if st.button("Read my question", type="primary", key="go_read"):
        q = ss.get("q_text", "")
        if ss.asked >= SESSION_LIMIT:
            st.error(f"You have reached the limit of {SESSION_LIMIT} questions for this visit. "
                     "Please come back later.")
        elif not q.strip():
            st.error("Please type a question first.")
        elif not take_daily_slot():
            st.error("This free tool has reached its daily limit. Please try again tomorrow.")
        else:
            ss.asked += 1
            with st.spinner("Reading your question..."):
                res = engine.parse_question(q, DEFAULT_GEO)
            if res["problem"]:
                st.error(res["problem"])
            else:
                ss.question, ss.parsed, ss.stage = q.strip(), res["steps"], "confirm"
                st.rerun()

# ---------------------------------------------------------------------------
# Stage: confirm how the question was read (and the distances)
# ---------------------------------------------------------------------------
elif ss.stage == "confirm":
    st.subheader("How I read your question")
    st.write(f"**Your question:** {esc(ss.question)}")
    for i, p in enumerate(ss.parsed):
        if not p["is_freight"]:
            st.markdown(f"**Step {i + 1}: {esc(p['label'])}** - {p['quantity']:g} {esc(p['unit'])}")
            continue
        if p["distance_only"]:
            st.markdown(f"**Step {i + 1}: {esc(p['label'])}** - a trip with no goods weight, so the factor "
                        "is per vehicle-km")
        else:
            st.markdown(f"**Step {i + 1}: {esc(p['label'])}** - {p['quantity']:g} {esc(p['unit'])} carried "
                        "over a distance (weight x distance)")
        if p["route_from"] and p["route_to"]:
            st.caption(f"Route: {p['route_from']} to {p['route_to']}")
        st.number_input(f"Distance for step {i + 1} (km)", min_value=0.0, step=1.0, format="%.1f",
                        value=float(p["distance_km"]) if p["distance_km"] else 0.0, key=f"km_{i}")
        how = p["distance_how"] or ""
        if how.startswith("AI estimate"):
            st.warning("This distance is an AI estimate and is unverified. Please check it (for example on a "
                       "map) and correct it if needed.")
        else:
            st.caption(f"Distance: {how}")

    c1, c2 = st.columns([1, 4])
    go = c1.button("Calculate", type="primary", key="go_calc")
    c2.button("Start over", key="restart_confirm", on_click=restart)
    if go:
        km_values = {i: float(ss.get(f"km_{i}") or 0) for i, p in enumerate(ss.parsed) if p["is_freight"]}
        run_steps, problem = engine.build_run_steps(ss.parsed, km_values)
        if problem:
            st.error(problem)
        else:
            with st.status("Matching emission factors (this can take a minute)...", expanded=True) as status:
                matches = engine.match_all(run_steps, lambda i, s: status.write(f"Step {i + 1}: {s['label']}"))
                status.update(label="Matching done", state="complete")
            ss.run_steps, ss.matches, ss.options, ss.chosen = run_steps, matches, {}, {}
            if any(engine.needs_stand_in(m) for m in matches):
                ss.stage = "standins"
            else:
                ss.final = engine.finalize(run_steps, matches, {})
                ss.stage = "result"
            st.rerun()

# ---------------------------------------------------------------------------
# Stage: stand-ins for steps without an exact factor
# ---------------------------------------------------------------------------
elif ss.stage == "standins":
    need = [i for i, m in enumerate(ss.matches) if engine.needs_stand_in(m)]
    st.subheader("No exact match for some steps")
    st.write("For these steps the database has no exact emission factor. The AI suggests the closest "
             "stand-ins it does have, and the numbers shown come from the database. You decide: leaving a "
             "step out is always allowed, and a stand-in you pick is labelled as yours in the result.")
    for i in need:
        if i not in ss.options:
            with st.spinner(f"Looking for the closest options for step {i + 1}..."):
                ss.options[i] = engine.stand_in_options(ss.run_steps[i], ss.matches[i]["pool"])
        step, info = ss.run_steps[i], ss.options[i]
        st.markdown(f"### Step {i + 1}: {esc(step['label'])}")
        if ss.matches[i].get("reasoning"):
            st.caption(f"Why there is no exact match: {ss.matches[i]['reasoning']}")
        if not info["options"]:
            st.info(f"{info['message']} This step stays an open gap.")
            continue
        labels = ["Leave this step out (open gap)"]
        for n, o in enumerate(info["options"], 1):
            st.markdown(
                f"**Option {n}: {esc(o['name'])}**  \n{esc(o['geography'])} - factor {o['factor_value']:.5g} "
                f"{esc(o['factor_unit'])} - would give **{fmt_kg(o['would_give'])} kg CO2e** for this step  \n"
                f"Close because: {esc(o['why_close'])}  \nDiffers: {esc(o['how_it_differs'])}  \n"
                f"Likely effect (AI opinion, unverified): {esc(o['likely_bias'])}")
            labels.append(f"Option {n}: {o['short_name']} ({fmt_kg(o['would_give'])} kg CO2e)")
        st.radio("Which stand-in, if any?", options=list(range(len(labels))), index=0,
                 format_func=lambda k, labels=labels: labels[k], key=f"pick_{i}")
        st.caption("Only pick one you could defend in a report.")

    if st.button("Show the result", type="primary", key="go_final"):
        chosen = {}
        for i in need:
            pick = ss.get(f"pick_{i}", 0)
            if pick and ss.options[i]["options"]:
                chosen[i] = ss.options[i]["options"][pick - 1]["ef_id"]
        ss.chosen = chosen
        ss.final = engine.finalize(ss.run_steps, ss.matches, chosen)
        ss.stage = "result"
        st.rerun()

# ---------------------------------------------------------------------------
# Stage: result
# ---------------------------------------------------------------------------
elif ss.stage == "result":
    results, summary = ss.final
    n_steps = summary["step_count"]
    n_gaps = len(summary["gaps"])
    covered = n_steps - n_gaps

    st.subheader("Result")
    st.write(f"**Your question:** {esc(ss.question)}")
    if covered == 0:
        st.error("No answer: not a single step could be calculated. This is not zero - the database has "
                 "no suitable factor for it.")
    else:
        c1, c2 = st.columns(2)
        c1.metric("Solid matches only",
                  f"{fmt_kg(summary['clean_total'])} kg CO2e" if summary["clean_count"] else "none",
                  help="Steps matched at high or medium confidence.")
        c1.caption(f"{summary['clean_count']} of {n_steps} steps")
        c2.metric("Including stand-ins and low-confidence steps", f"{fmt_kg(summary['grand_total'])} kg CO2e")
        c2.caption(f"Covers {covered} of {n_steps} steps")
    if n_gaps:
        st.warning(f"{n_gaps} step(s) have no emission factor and are NOT included (they are missing, not "
                   "zero), so the totals understate the footprint.")

    st.markdown("#### Step by step, with sources")
    st.dataframe(pd.DataFrame(engine.table_rows(results)), hide_index=True,
                 column_config={"kg CO2e": st.column_config.NumberColumn(format="%.4f")})

    for i, r in enumerate(results, 1):
        with st.expander(f"Show working for step {i}: {r['step']}"):
            d = r.get("detail")
            if r["status"] in ("no_match", "cannot_calculate"):
                st.write(f"No answer for this step: {r['reason']}")
                st.caption("Nothing was guessed, and it is not counted as zero.")
                continue
            st.markdown(f"**Factor:** {esc(r['activity'])} ({esc(d['category'])} / {esc(d['variant'] or '-')}), "
                        f"{esc(d['geography'])}  \n**Value:** {r['factor_value']:.6g} {esc(r['factor_unit'])}")
            st.markdown(f"**Working:** {esc(r['conversion_note'])}; {r['converted']:.6g} x "
                        f"{r['factor_value']:.6g} = **{r['contribution']:.4f} kg CO2e**")
            src = [f"**Source:** {esc(d['source_name'])}"]
            if d.get("publisher"):
                src.append(f"Publisher: {esc(d['publisher'])}")
            ver = " / ".join(str(x) for x in (d.get("source_version"), d.get("source_year")) if x)
            src.append(f"Version / year: {esc(ver) if ver else 'not recorded in the database'}")
            if d.get("license_type"):
                src.append(f"Licence: {esc(d['license_type'])}")
            if d.get("source_url"):
                src.append(f"Link: [{d['source_url']}]({d['source_url']})")
            if d.get("study_year"):
                src.append(f"Underlying study year: {d['study_year']}")
            ref = ", ".join(str(x) for x in (d.get("source_sheet"), d.get("source_row_ref")) if x)
            src.append(f"Database row: {d['ef_id']}" + (f" (source reference: {esc(ref)})" if ref else ""))
            st.markdown("  \n".join(src))
            if r.get("ai_reasoning") and not r.get("user_proxy"):
                st.markdown(f"**Why the AI chose it ({esc(r['confidence'])} confidence):** {esc(r['ai_reasoning'])}")
            for note in r["notes"]:
                st.caption(f"Note: {note}")

    st.markdown("#### Sources used in this result")
    for entry in engine.sources_used(results):
        lines = engine.source_markdown(entry)
        body = ["**" + esc(lines[0]) + "**"] + [esc(x) for x in lines[1:] if not x.startswith("Link:")]
        body.append(f"Link: [{entry['url']}]({entry['url']})" if entry.get("url")
                    else "Link: not recorded in the database")
        st.markdown("  \n".join(body))

    assumptions = [f"Distance for step {i}: {r['trip_km']:g} km ({r['distance_note']})"
                   for i, r in enumerate(results, 1) if r.get("distance_note")]
    if assumptions:
        st.markdown("#### Assumptions to check")
        for a in assumptions:
            st.write("- " + a)

    st.markdown("#### Save this result")
    d1, d2 = st.columns(2)
    d1.download_button("Download CSV (opens in Excel)", engine.to_csv(ss.question, results, summary),
                       file_name="carbon_result.csv", mime="text/csv", key="dl_csv")
    d2.download_button("Download report (open it, then print or save as PDF)",
                       engine.to_html(ss.question, results, summary),
                       file_name="carbon_result.html", mime="text/html", key="dl_html")
    st.button("Ask another question", key="again", on_click=restart)