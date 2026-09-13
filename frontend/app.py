"""Streamlit UI: ticker input, live debate feed (Bull/Bear/Judge message bubbles),
deep-dive buttons per filing section, final memo card with verdict and fact-check
count.

Talks directly to orchestration.debate_loop and deep_dive.spawner in-process —
no FastAPI/Redis/worker layer yet, that's Phase 10's job (progress.md's
architecture diagram scopes the queue/worker pattern as "added later" for
deployment, not required for a working single-user UI).

Run with:
    streamlit run frontend/app.py
"""

import sys
from pathlib import Path

# `streamlit run` puts this file's own directory (frontend/) on sys.path, not the
# project root, so sibling packages (orchestration, deep_dive, agents, ...) would
# otherwise fail to import.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.console import fix_windows_console

# Streamlit executes module-level code (including st.set_page_config below)
# before main() ever runs, so this must happen here, not inside main().
fix_windows_console()

import streamlit as st
from dotenv import load_dotenv
from litellm.exceptions import RateLimitError

from deep_dive.spawner import get_available_sections, spawn_deep_dive
from orchestration.debate_loop import run_debate_stream

load_dotenv()

st.set_page_config(page_title="Equity Research Debate Desk", page_icon="⚖️")

AVATARS = {"bull": "🐂", "bear": "🐻", "judge": "⚖️"}
DEFAULT_ROUNDS = 3


def _render_statement(entry: dict) -> None:
    with st.chat_message(entry["agent"], avatar=AVATARS.get(entry["agent"], "💬")):
        st.markdown(f"**{entry['agent'].upper()} — Round {entry['round']}**")
        st.write(entry["statement"])


def _render_memo_card(judge_verdict: dict) -> None:
    with st.container(border=True):
        st.subheader("Judge's verdict")
        st.markdown(f"**Stronger side:** {judge_verdict['stronger_side'].upper()}")
        st.write(judge_verdict["memo"])
        checked, unsupported = judge_verdict["claims_checked"], judge_verdict["claims_unsupported"]
        supported = checked - unsupported
        st.caption(f"Fact-check: {supported}/{checked} claims supported ({unsupported} unsupported or not enough evidence)")


def _render_deep_dive(company: str, debate_result: dict) -> None:
    st.subheader("Deep dive")
    if "deep_dive_sections" not in st.session_state:
        st.session_state.deep_dive_sections = get_available_sections(company)

    for section in st.session_state.deep_dive_sections:
        existing = debate_result["deep_dive_results"].get(section)
        clicked = st.button(section, key=f"deep_dive_{section}")
        if clicked or existing:
            with st.spinner(f"Summarizing {section!r}...") if clicked and not existing else st.container():
                if clicked and not existing:
                    spawn_deep_dive(company, section, debate_result=debate_result)
                    existing = debate_result["deep_dive_results"][section]
            if existing:
                with st.container(border=True):
                    st.markdown(f"**{section}**")
                    st.write(existing["summary"])


def main() -> None:
    st.title("Equity Research Debate Desk")
    st.caption("Bull vs Bear debate a company's real SEC filing, then a Judge scores it.")

    company = st.text_input("Company ticker", value="AAPL").strip().upper()
    run_clicked = st.button("Run debate", type="primary")

    if run_clicked:
        st.session_state.pop("debate_result", None)
        st.session_state.pop("deep_dive_sections", None)
        st.session_state.company = company

        feed = st.container()
        status_placeholder = st.empty()

        def on_wait(remaining_seconds: float, reason: str) -> None:
            status_placeholder.warning(
                f"⏳ Groq's rate limit was hit during **{reason}** — "
                f"waiting {remaining_seconds:.0f}s before retrying..."
            )

        try:
            with st.spinner(f"Indexing {company}'s filing..."):
                stream = run_debate_stream(company, rounds=DEFAULT_ROUNDS, on_wait=on_wait)
                first_event = next(stream)
            st.toast(f"Indexed {first_event['chunks_indexed']} chunks.")

            for event in stream:
                if event["type"] == "statement":
                    status_placeholder.empty()
                    with feed:
                        _render_statement(event["entry"])
                elif event["type"] == "done":
                    st.session_state.debate_result = event["result"]
        except (RuntimeError, RateLimitError):
            st.error(
                "Groq's rate limit was hit and retries were exhausted. Wait a bit "
                "and try again, or check if you've hit the daily free-tier cap."
            )
        finally:
            status_placeholder.empty()

    if "debate_result" in st.session_state and st.session_state.get("company") == company:
        result = st.session_state.debate_result
        if not run_clicked:
            feed = st.container()
            with feed:
                for entry in result["transcript"]:
                    _render_statement(entry)
        _render_memo_card(result["judge_verdict"])
        _render_deep_dive(company, result)


if __name__ == "__main__":
    main()