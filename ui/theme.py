"""Page styling and shared layout pieces."""

from __future__ import annotations

import html

import streamlit as st

from .state import STEPS

_CSS = """
<style>
.block-container {padding-top: 2.2rem; max-width: 1240px;}
.rtd-hero h1 {font-size: 2.1rem; margin-bottom: .2rem;}
.rtd-hero p {color: #94a3b8; font-size: 1.05rem; margin-top: 0;}
.rtd-card {border: 1px solid rgba(148,163,184,.25); border-radius: 12px; padding: 1rem 1.1rem;
           background: rgba(15,23,42,.35); height: 100%;}
.rtd-card h4 {margin: 0 0 .35rem 0; font-size: 1rem;}
.rtd-card p {margin: 0; color: #94a3b8; font-size: .92rem;}
.rtd-steps {display: flex; gap: .5rem; flex-wrap: wrap; margin: .4rem 0 1.2rem;}
.rtd-step {padding: .35rem .8rem; border-radius: 999px; font-size: .85rem;
           border: 1px solid rgba(148,163,184,.3); color: #94a3b8;}
.rtd-step.done {color: #2dd4bf; border-color: rgba(45,212,191,.5);}
.rtd-step.active {background: #14b8a6; color: #04111d; border-color: #14b8a6; font-weight: 600;}
.rtd-chips {display: flex; gap: .5rem; flex-wrap: wrap; margin: .2rem 0 .8rem;}
.rtd-chip {border: 1px solid rgba(148,163,184,.3); border-radius: 8px; padding: .3rem .7rem; font-size: .85rem;}
.rtd-chip span {color: #94a3b8; margin-right: .35rem;}
.rtd-note {border-left: 3px solid #60a5fa; padding: .55rem .9rem; background: rgba(96,165,250,.08);
           border-radius: 0 8px 8px 0; color: #cbd5e1; font-size: .9rem;}
</style>
"""


def apply() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


def step_bar(current: int) -> None:
    pills = []
    for index, name in enumerate(STEPS):
        css = "active" if index == current else "done" if index < current else ""
        pills.append(f'<span class="rtd-step {css}">{index + 1}. {name}</span>')
    st.markdown(f'<div class="rtd-steps">{"".join(pills)}</div>', unsafe_allow_html=True)


def card(title: str, body: str) -> None:
    st.markdown(f'<div class="rtd-card"><h4>{title}</h4><p>{body}</p></div>', unsafe_allow_html=True)


def note(text: str) -> None:
    st.markdown(f'<div class="rtd-note">{text}</div>', unsafe_allow_html=True)


def chips(items: dict[str, object]) -> None:
    body = "".join(
        f'<div class="rtd-chip"><span>{html.escape(str(k))}</span>{html.escape(str(v))}</div>' for k, v in items.items()
    )
    st.markdown(f'<div class="rtd-chips">{body}</div>', unsafe_allow_html=True)
