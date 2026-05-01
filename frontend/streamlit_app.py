"""
Agentic Multimodal RAG — Streamlit Frontend
Claude-style dark UI with sidebar nav, right-aligned user messages, bottom input.
"""
from __future__ import annotations
import uuid
import os
import time
import httpx
import streamlit as st

# ── Constants ───────────────────────────────────────────────────────────────────
API_BASE = os.environ.get("API_BASE", "http://localhost:8000").rstrip("/")
# For browser redirects (OAuth), use a host-reachable URL, not Docker-internal DNS.
BROWSER_API_BASE = os.environ.get("BROWSER_API_BASE", API_BASE).rstrip("/")
GOOGLE_OAUTH_URL = f"{BROWSER_API_BASE}/auth/google/login"

st.set_page_config(
    page_title="RAG Assistant",
    page_icon="✳️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS — Claude-inspired ───────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Styrene+A:wght@400;500;600&family=Inter:wght@400;500;600&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stApp"] {
    background: #1a1a1a !important;
    color: #ececec !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

/* ── Hide all Streamlit chrome ── */
#MainMenu, footer, header,
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"],
.stDeployButton { display: none !important; }

[data-testid="collapsedControl"] { display: none !important; }

.main .block-container {
    padding: 0 !important;
    max-width: 100% !important;
}

/* ── SIDEBAR — Claude left nav ── */
[data-testid="stSidebar"] {
    background: #1a1a1a !important;
    border-right: 1px solid rgba(255,255,255,0.08) !important;
    width: 260px !important;
    min-width: 260px !important;
}
[data-testid="stSidebar"] > div:first-child {
    padding: 0 !important;
}
[data-testid="stSidebar"] .block-container {
    padding: 0 !important;
}

/* Sidebar text / labels */
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stCaption,
[data-testid="stSidebar"] small { color: #8a8a8a !important; font-size: 13px !important; }

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #ececec !important; }

/* Sidebar nav buttons */
[data-testid="stSidebar"] .stButton > button {
    background: transparent !important;
    color: #adadad !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 8px 12px !important;
    font-size: 14px !important;
    font-weight: 400 !important;
    text-align: left !important;
    width: 100% !important;
    cursor: pointer !important;
    transition: background 0.15s, color 0.15s !important;
    text-decoration: none !important;
    display: flex !important;
    align-items: center !important;
    gap: 10px !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(255,255,255,0.06) !important;
    color: #ececec !important;
    border: none !important;
    text-decoration: none !important;
}
[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.07) !important;
    margin: 6px 0 !important;
}
/* Active nav button highlight */
[data-testid="stSidebar"] .stButton > button:disabled {
    background: rgba(204,120,92,0.15) !important;
    color: #cc785c !important;
    opacity: 1 !important;
    cursor: default !important;
}
/* ── Chat list rows ── */
/* Title buttons: left-aligned */
[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] [data-testid="column"]:first-child .stButton > button {
    justify-content: flex-start !important;
    text-align: left !important;
    padding: 0 10px !important;
    height: 36px !important;
    min-height: 36px !important;
    font-size: 13px !important;
    white-space: nowrap !important;
    overflow: hidden !important;
}
[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] [data-testid="column"]:first-child .stButton > button p,
[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] [data-testid="column"]:first-child .stButton > button span,
[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] [data-testid="column"]:first-child .stButton > button div {
    text-align: left !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
    width: 100% !important;
    margin: 0 !important;
}
/* Delete button beside each chat title */
[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] [data-testid="column"]:last-child {
    width: 32px !important;
    min-width: 32px !important;
    max-width: 32px !important;
    flex: 0 0 32px !important;
}
[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] [data-testid="column"]:last-child .stButton > button,
[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] [data-testid="column"]:last-child > div {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    color: #777 !important;
    font-size: 21px !important;
    font-weight: 500 !important;
    letter-spacing: 0 !important;
    padding: 0 !important;
    width: 32px !important;
    min-width: 32px !important;
    max-width: 32px !important;
    height: 32px !important;
    min-height: 32px !important;
    max-height: 32px !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    border-radius: 6px !important;
    line-height: 1 !important;
    margin: 0 auto !important;
}
[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] [data-testid="column"]:last-child .stButton > button:hover {
    color: #ececec !important;
    background: rgba(255,255,255,0.08) !important;
}
/* ── AUTH PAGE ── */
.auth-wrap {
    display: flex;
    align-items: center;
    justify-content: center;
    min-height: 100vh;
    background: #1a1a1a;
}
.auth-card {
    width: 420px;
    background: #232323;
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 40px 36px 32px;
    box-shadow: 0 20px 60px rgba(0,0,0,0.6);
}
.auth-logo {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 28px;
}
.auth-logo-icon {
    width: 36px; height: 36px;
    background: #cc785c;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 28px;
}
.auth-logo-name {
    font-size: 20px;
    font-weight: 600;
    color: #ececec;
    letter-spacing: -0.3px;
}
.auth-heading {
    font-size: 22px;
    font-weight: 600;
    color: #ececec;
    margin-bottom: 6px;
    letter-spacing: -0.3px;
}
.auth-sub {
    font-size: 13px;
    color: #8a8a8a;
    margin-bottom: 24px;
}

/* Google button */
.g-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 10px;
    width: 100%;
    padding: 11px 16px;
    background: #fff;
    color: #1a1a1a;
    font-size: 14px;
    font-weight: 600;
    border-radius: 8px;
    border: none;
    cursor: pointer;
    margin-bottom: 16px;
    transition: background 0.15s, box-shadow 0.15s;
    font-family: 'Inter', sans-serif;
    text-decoration: none;
}
.g-btn:hover { background: #f0f0f0; box-shadow: 0 2px 8px rgba(0,0,0,0.2); }
.g-btn:disabled { opacity: 0.7; cursor: default; }

.or-row {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 0 0 16px;
    color: #555;
    font-size: 12px;
    letter-spacing: 0.5px;
}
.or-row::before, .or-row::after {
    content: "";
    flex: 1;
    height: 1px;
    background: rgba(255,255,255,0.08);
}
.toggle-row {
    text-align: center;
    margin-top: 16px;
    font-size: 13px;
    color: #666;
}

/* Input fields */
div[data-baseweb="input"] {
    background: #2a2a2a !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 8px !important;
}
div[data-baseweb="input"]:focus-within {
    border-color: #cc785c !important;
    box-shadow: 0 0 0 3px rgba(204,120,92,0.15) !important;
}
div[data-baseweb="input"] input {
    background: transparent !important;
    color: #ececec !important;
    font-size: 14px !important;
    font-family: 'Inter', sans-serif !important;
    padding: 10px 14px !important;
}
div[data-baseweb="input"] input::placeholder { color: #555 !important; }
div[data-baseweb="input"] label { display: none !important; }
[data-testid="stTextInput"] > label {
    font-size: 13px !important;
    color: #8a8a8a !important;
    font-weight: 400 !important;
    margin-bottom: 4px !important;
}

/* Submit button */
[data-testid="stFormSubmitButton"] button {
    width: 100% !important;
    background: #cc785c !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-size: 14px !important;
    font-weight: 600 !important;
    padding: 11px 0 !important;
    margin-top: 4px !important;
    cursor: pointer !important;
    transition: opacity 0.15s !important;
    font-family: 'Inter', sans-serif !important;
    letter-spacing: -0.1px !important;
}
[data-testid="stFormSubmitButton"] button:hover { opacity: 0.88 !important; }

/* Toggle link button */
.stButton > button {
    background: transparent !important;
    color: #cc785c !important;
    border: none !important;
    padding: 0 !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    cursor: pointer !important;
    text-decoration: none !important;
    font-family: 'Inter', sans-serif !important;
}
.stButton > button:hover {
    color: #e08a6a !important;
    background: transparent !important;
    border: none !important;
    text-decoration: underline !important;
}

/* ── CHAT PAGE ── */

/* + button injected into chat input (cosmetic) */
.chat-plus-btn {
    position: absolute !important;
    left: 10px !important;
    bottom: 10px !important;
    width: 30px !important; height: 30px !important;
    background: transparent !important;
    color: #8a8a8a !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 50% !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    font-size: 18px !important;
    line-height: 1 !important;
    cursor: pointer !important;
    transition: background 0.15s, color 0.15s !important;
    z-index: 10 !important;
    user-select: none !important;
}
.chat-plus-btn:hover {
    background: rgba(255,255,255,0.06) !important;
    color: #ececec !important;
}


/* ── Chat message rows — base reset ── */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
    margin: 0 auto 28px !important;
    max-width: 680px !important;
    display: flex !important;
    align-items: flex-start !important;
    gap: 12px !important;
}

/* ── USER: row-reverse so bubble goes RIGHT ── */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    flex-direction: row-reverse !important;
}

/* User avatar — small circle on far right */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"])
  [data-testid="chatAvatarIcon-user"] {
    width: 28px !important; height: 28px !important; min-width: 28px !important;
    border-radius: 50% !important;
    background: #cc785c !important;
    color: #fff !important;
    font-size: 12px !important; font-weight: 700 !important;
    display: flex !important; align-items: center !important; justify-content: center !important;
    flex-shrink: 0 !important; margin-top: 4px !important;
}

/* User bubble — dark pill on the right */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"])
  [data-testid="stChatMessageContent"],
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) > div:nth-child(2) {
    background: #242424 !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 20px 4px 20px 20px !important;
    padding: 10px 16px !important;
    max-width: 65% !important;
    color: #e8e8e8 !important;
    font-size: 15px !important;
    line-height: 1.6 !important;
}

/* ── ASSISTANT: plain text on the LEFT, no bubble ── */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
    flex-direction: row !important;
}

/* Assistant avatar — small ✳️ icon */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"])
  [data-testid="chatAvatarIcon-assistant"] {
    width: 28px !important; height: 28px !important; min-width: 28px !important;
    border-radius: 50% !important;
    background: #1e1e1e !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    color: #cc785c !important;
    font-size: 14px !important;
    display: flex !important; align-items: center !important; justify-content: center !important;
    flex-shrink: 0 !important; margin-top: 4px !important;
}

/* Assistant content — plain, no background */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"])
  [data-testid="stChatMessageContent"],
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) > div:nth-child(2) {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
    max-width: 100% !important;
    color: #e0e0e0 !important;
    font-size: 15px !important;
    line-height: 1.8 !important;
}

/* Caption under assistant (badge + latency) */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"])
  .stCaptionContainer,
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"])
  [data-testid="stCaptionContainer"] {
    color: #4a4a4a !important;
    font-size: 11px !important;
    margin-top: 6px !important;
}

/* Sources expander */
[data-testid="stExpander"] {
    background: rgba(255,255,255,0.02) !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: 8px !important;
    margin-top: 10px !important;
}
[data-testid="stExpander"] summary,
[data-testid="stExpander"] [data-testid="stExpanderToggleIcon"] {
    color: #555 !important;
    font-size: 12px !important;
}

/* ── Chat input — sticky bottom, blended into the page ── */
[data-testid="stChatInput"] {
    background: #1f1f1f !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 18px !important;
    max-width: 760px !important;
    margin: 0 auto !important;
    min-height: 68px !important;
    padding: 2px 4px 4px !important;
    box-shadow: 0 4px 24px rgba(0,0,0,0.25) !important;
    position: relative !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: rgba(255,255,255,0.16) !important;
    box-shadow: 0 4px 24px rgba(0,0,0,0.3) !important;
}
[data-testid="stChatInput"] textarea {
    background: transparent !important;
    color: #ececec !important;
    font-size: 15px !important;
    font-family: 'Inter', sans-serif !important;
    caret-color: #cc785c !important;
    padding: 10px 14px !important;
    padding-right: 96px !important;   /* room for mic + submit */
    min-height: 42px !important;
}
[data-testid="stChatInput"] textarea::placeholder { color: #6a6a6a !important; }
[data-testid="stChatInput"] textarea::placeholder { color: #555 !important; }

/* Submit button — pinned to bottom-right of the input pill */
[data-testid="stChatInputSubmitButton"] {
    position: absolute !important;
    bottom: 12px !important;
    right: 12px !important;
}
[data-testid="stChatInputSubmitButton"] > button {
    background: #cc785c !important;
    border-radius: 8px !important;
    width: 36px !important; height: 36px !important;
    color: #fff !important;
}
[data-testid="stChatInputSubmitButton"] > button:hover {
    background: #b8674d !important;
}

/* stBottom bar — opaque solid, sits in its own reserved strip. Since
   stMain above it is height-capped, there's a hard cutoff between the
   scrolling chat area and the input bar — no text ever appears within
   or below the pill. */
[data-testid="stBottomBlockContainer"],
[data-testid="stBottom"] > div,
[data-testid="stBottom"] section,
[data-testid="stBottom"] [data-testid="stVerticalBlock"] {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}
[data-testid="stBottom"] {
    background: #1a1a1a !important;
    padding: 6px 24px 10px !important;
    position: sticky !important;
    bottom: 0 !important;
    left: 0 !important;
    right: 0 !important;
    z-index: 120 !important;
    pointer-events: auto !important;
}
[data-testid="stBottom"] > div {
    max-width: 760px !important;
    margin: 0 auto !important;
    position: relative !important;
    pointer-events: auto !important;
}

/* ── Mic button: fixed just left of the submit button ── */
.mic-fixed-wrap {
    position: fixed !important;
    bottom: 20px !important;
    right: max(32px, calc(50vw + 130px - 370px + 48px)) !important;
    z-index: 9999 !important;
    width: 40px !important;
    height: 40px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}
/* The chat scroll container: Streamlit renders st.container(height=612)
   as a div with inline `height: 612px; overflow: auto`. We match that
   via attribute selector (612 is a unique sentinel) and stretch it to
   fill the viewport minus the input strip. Content scrolls entirely
   within this container — there is NO way for text to reach the pill. */
[data-testid="stVerticalBlockBorderWrapper"][style*="height: 612px"],
[data-testid="stVerticalBlockBorderWrapper"][style*="height:612px"] {
    height: calc(100vh - 120px) !important;
    max-height: calc(100vh - 120px) !important;
    padding-bottom: 28px !important;
}

/* Hide the mic wrapper from normal flow — it's repositioned by JS */
.mic-fixed-wrap {
    visibility: hidden;
    pointer-events: none;
    width: 0 !important;
    height: 0 !important;
}
/* JS will make it visible once positioned */
.mic-fixed-wrap.mic-ready {
    visibility: visible !important;
    pointer-events: auto !important;
    width: 40px !important;
    height: 40px !important;
}

/* AGGRESSIVE: hide the audio_recorder iframe's element-container until JS
   marks it positioned. The wrapper div above doesn't actually contain the
   iframe (Streamlit renders components in their own slot), so the iframe
   was leaking into the welcome screen as an empty black bar. */
[data-testid="element-container"]:has(iframe[title*="audio_recorder"]):not(.mic-positioned),
[data-testid="stIFrame"]:has(iframe[title*="audio_recorder"]):not(.mic-positioned) {
    position: fixed !important;
    bottom: -9999px !important;
    left: -9999px !important;
    width: 40px !important;
    height: 40px !important;
    margin: 0 !important;
    padding: 0 !important;
    overflow: hidden !important;
}

/* Once positioned: orange pill matching the submit button */
[data-testid="element-container"].mic-positioned,
[data-testid="stIFrame"].mic-positioned {
    background: #cc785c !important;
    border-radius: 8px !important;
    transition: background 0.15s !important;
}
[data-testid="element-container"].mic-positioned:hover,
[data-testid="stIFrame"].mic-positioned:hover {
    background: #b8674d !important;
}
[data-testid="element-container"].mic-positioned iframe,
[data-testid="stIFrame"].mic-positioned iframe {
    background: transparent !important;
    border-radius: 8px !important;
}

/* Expander (sources) */
[data-testid="stExpander"] {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 8px !important;
}
[data-testid="stExpander"] summary { color: #8a8a8a !important; font-size: 13px !important; }

/* Alerts */
[data-testid="stAlert"] { border-radius: 8px !important; font-size: 13px !important; }

/* Spinner */
[data-testid="stSpinner"] { color: #cc785c !important; }

/* Sidebar brand area */
.sidebar-brand {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 18px 16px 12px;
    font-size: 16px;
    font-weight: 600;
    color: #ececec;
    letter-spacing: -0.2px;
    border-bottom: 1px solid rgba(255,255,255,0.07);
    margin-bottom: 6px;
}
.sidebar-brand-icon {
    width: 28px; height: 28px;
    background: #cc785c;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 14px;
    flex-shrink: 0;
}
.sidebar-section {
    padding: 8px 16px 4px;
    font-size: 11px;
    font-weight: 600;
    color: #555;
    letter-spacing: 0.8px;
    text-transform: uppercase;
}
.sidebar-user {
    padding: 12px 16px;
    border-top: 1px solid rgba(255,255,255,0.07);
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 13px;
    color: #8a8a8a;
}
.sidebar-user-avatar {
    width: 28px; height: 28px;
    background: #3a3a3a;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 12px;
    font-weight: 600;
    color: #ececec;
    flex-shrink: 0;
}

/* Welcome screen */
.welcome-wrap {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-height: 55vh;
    padding: 60px 24px 40px;
    text-align: center;
    max-width: 680px;
    margin: 0 auto;
}
.welcome-icon {
    width: 60px; height: 60px;
    background: #cc785c;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 28px;
    margin-bottom: 24px;
}
.welcome-title {
    font-size: 28px;
    font-weight: 600;
    color: #ececec;
    letter-spacing: -0.5px;
    margin-bottom: 10px;
}
.welcome-sub {
    font-size: 15px;
    color: #6a6a6a;
    max-width: 400px;
    line-height: 1.65;
}

/* ── Upload page file uploader ── */
[data-testid="stMain"] [data-testid="stFileUploader"] section {
    background: rgba(255,255,255,0.03) !important;
    border: 2px dashed rgba(255,255,255,0.12) !important;
    border-radius: 12px !important;
    padding: 28px 20px !important;
    min-height: 120px !important;
    transition: border-color 0.2s, background 0.2s !important;
}
[data-testid="stMain"] [data-testid="stFileUploader"] section:hover {
    border-color: rgba(204,120,92,0.4) !important;
    background: rgba(255,255,255,0.04) !important;
}
[data-testid="stMain"] [data-testid="stFileUploaderFile"] {
    display: none !important;
}
[data-testid="stMain"] [data-testid="stFileUploaderDropzoneInstructions"] {
    color: #8a8a8a !important;
}

/* Go-to-chat button on upload page */
button[data-testid="baseButton-secondary"][kind="secondary"] {
    transition: background 0.15s !important;
}
</style>
""", unsafe_allow_html=True)


# ── Session state ───────────────────────────────────────────────────────────────
def _init_state():
    defaults = {
        "token": None,
        "username": None,
        "user_id": None,
        "session_id": str(uuid.uuid4()),
        "messages": [],
        "auth_mode": "login",
        "page": "chat",       # "chat" or "upload"
        "is_new_user": False,
        "chats": {},          # {session_id: {"title": str, "messages": list}}
        "chats_loaded": False,
        "mic_counter": 0,
        "upload_nonce": 0,
        "upload_notice": None,
        "upload_pending": False,
        "_renaming_sid": None,
        "chat_menu_open": None,
        "_last_api_error": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _load_persisted_chats():
    """Fetch all chat sessions from the backend and populate st.session_state.chats."""
    resp = api_get("/chat/sessions", timeout=5.0)
    if resp is None or resp.status_code != 200:
        st.session_state.chats_loaded = True
        return
    sessions = resp.json().get("sessions", [])
    chats: dict = {}
    for s in sessions:
        sid = s["session_id"]
        chats[sid] = {
            "title": s.get("title", "Chat"),
            "messages": None,   # lazy-loaded on click
        }
    st.session_state.chats = chats
    st.session_state.chats_loaded = True


def _load_session_messages(session_id: str) -> list[dict]:
    """Fetch messages for a specific session from the backend."""
    resp = api_get(f"/chat/sessions/{session_id}", timeout=10.0)
    if resp is None or resp.status_code != 200:
        return []
    raw = resp.json().get("messages", [])
    messages = []
    for m in raw:
        entry = {"role": m["role"], "content": m["content"]}
        if m.get("sources"):
            try:
                import json
                entry["sources"] = json.loads(m["sources"])
            except Exception:
                pass
        messages.append(entry)
    return messages


def _register_current_chat(first_message: str) -> None:
    """Register the active session at the TOP of sidebar history on first message."""
    sid = st.session_state.session_id
    if sid in st.session_state.chats:
        return
    title = first_message.strip().split("\n", 1)[0][:40]
    if len(first_message) > 40:
        title += "…"
    new_entry = {sid: {
        "title": title or "New chat",
        "messages": st.session_state.messages,
    }}
    new_entry.update(st.session_state.chats)
    st.session_state.chats = new_entry

_init_state()


# ── API helpers ─────────────────────────────────────────────────────────────────
def _headers():
    return {"Authorization": f"Bearer {st.session_state.token}"}

def api_call(method: str, path: str, **kwargs):
    try:
        response = getattr(httpx, method)(f"{API_BASE}{path}", **kwargs)
        st.session_state["_last_api_error"] = ""
        return response
    except Exception as exc:
        st.session_state["_last_api_error"] = str(exc)
        return None

def api_post(path: str, json=None, files=None, auth=True, timeout: float = 30.0):
    return api_call("post", path, json=json, files=files,
                    headers=_headers() if auth else {}, timeout=timeout)

def api_get(path: str, timeout: float = 5.0):
    return api_call("get", path, headers=_headers(), timeout=timeout)

def api_delete(path: str):
    return api_call("delete", path, headers=_headers(), timeout=10)

def api_patch(path: str, json=None):
    return api_call("patch", path, json=json, headers=_headers(), timeout=10)


def _api_post_with_retry(path: str, *, json=None, files=None, auth=True, timeout: float = 30.0):
    """
    Retry once for transient network startup hiccups so first upload doesn't fail
    when backend is still warming up.
    """
    resp = api_post(path, json=json, files=files, auth=auth, timeout=timeout)
    if resp is not None:
        return resp
    err = (st.session_state.get("_last_api_error") or "").lower()
    transient_markers = (
        "connection refused",
        "connecterror",
        "server disconnected",
        "temporarily unavailable",
        "timed out",
        "timeout",
    )
    if any(marker in err for marker in transient_markers):
        time.sleep(1.0)
        return api_post(path, json=json, files=files, auth=auth, timeout=timeout)
    return None

def safe_json(resp, fallback: str = "Unknown error") -> str:
    try:
        return resp.json().get("detail", fallback)
    except Exception:
        return resp.text[:200] if resp.text else fallback


def _mark_upload_pending() -> None:
    st.session_state.upload_pending = True


# ── Google OAuth token pickup ───────────────────────────────────────────────────
def _pick_up_google_token():
    params = st.query_params
    if "g_token" in params:
        st.session_state.token      = params["g_token"]
        st.session_state.username   = params.get("g_user", "user")
        try:
            st.session_state.user_id = int(params.get("g_uid", 0))
        except ValueError:
            st.session_state.user_id = 0
        st.session_state.messages   = []
        st.session_state.session_id = str(uuid.uuid4())
        st.query_params.clear()
        _load_persisted_chats()
        st.session_state.page = "chat"
        st.rerun()

_pick_up_google_token()


# ── AUTH PAGE ───────────────────────────────────────────────────────────────────
def render_auth():
    _, col, _ = st.columns([1, 1.1, 1])
    with col:
        # Logo + heading
        mode = st.session_state.auth_mode
        heading = "Welcome back" if mode == "login" else "Create your account"
        sub = "Sign in to continue" if mode == "login" else "Get started for free"

        st.markdown(f"""
        <div style="margin-bottom:28px;">
            <div style="display:flex;align-items:center;gap:10px;margin-bottom:20px;">
                <div style="width:36px;height:36px;background:#cc785c;border-radius:50%;
                     display:flex;align-items:center;justify-content:center;font-size:18px;">✳️</div>
                <span style="font-size:20px;font-weight:600;color:#ececec;">RAG Assistant</span>
            </div>
            <div style="font-size:22px;font-weight:600;color:#ececec;letter-spacing:-0.3px;
                 margin-bottom:4px;">{heading}</div>
            <div style="font-size:13px;color:#8a8a8a;">{sub}</div>
        </div>
        """, unsafe_allow_html=True)

        # Google OAuth button (same-window redirect)
        google_html = f"""
        <a class="g-btn" href="{GOOGLE_OAUTH_URL}" target="_self">
            <svg width="17" height="17" viewBox="0 0 24 24" style="flex-shrink:0">
              <path d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z" fill="#4285F4"/>
              <path d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z" fill="#34A853"/>
              <path d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l3.66-2.84z" fill="#FBBC05"/>
              <path d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z" fill="#EA4335"/>
            </svg>
            Continue with Google
        </a>
        """
        st.markdown(google_html, unsafe_allow_html=True)

        st.markdown('<div class="or-row">OR</div>', unsafe_allow_html=True)

        # ── Login form ──────────────────────────────────────────────────────────
        if mode == "login":
            with st.form("login_form", clear_on_submit=False):
                username  = st.text_input("Username", placeholder="Your username")
                password  = st.text_input("Password", placeholder="Your password",
                                          type="password")
                submitted = st.form_submit_button("Sign in →", use_container_width=True)

            if submitted:
                if not username or not password:
                    st.error("Please fill in both fields.")
                else:
                    with st.spinner("Signing in…"):
                        resp = api_post("/auth/login",
                                        json={"username": username, "password": password},
                                        auth=False)
                    if resp is None:
                        st.error("Cannot reach the server — is the backend running?")
                    elif resp.status_code == 200:
                        d = resp.json()
                        st.session_state.token      = d["access_token"]
                        st.session_state.username   = d["username"]
                        st.session_state.user_id    = d["user_id"]
                        st.session_state.messages   = []
                        st.session_state.session_id = str(uuid.uuid4())
                        _load_persisted_chats()
                        st.session_state.page       = "chat"
                        st.session_state.is_new_user = False
                        st.rerun()
                    else:
                        st.error(safe_json(resp, "Incorrect username or password."))

            st.markdown('<div class="toggle-row">No account?&nbsp;</div>',
                        unsafe_allow_html=True)
            if st.button("Create one →", key="go_register"):
                st.session_state.auth_mode = "register"
                st.rerun()

        # ── Register form ───────────────────────────────────────────────────────
        else:
            with st.form("register_form", clear_on_submit=False):
                r_user   = st.text_input("Username", placeholder="Pick a username")
                r_email  = st.text_input("Email",    placeholder="you@example.com")
                r_pass   = st.text_input("Password", placeholder="At least 6 characters",
                                         type="password")
                r_conf   = st.text_input("Confirm password", placeholder="Repeat password",
                                         type="password")
                reg_sub  = st.form_submit_button("Create account →", use_container_width=True)

            if reg_sub:
                if not r_user or not r_email or not r_pass:
                    st.error("Please fill in all fields.")
                elif r_pass != r_conf:
                    st.error("Passwords do not match.")
                elif len(r_pass) < 6:
                    st.error("Password must be at least 6 characters.")
                else:
                    with st.spinner("Creating account…"):
                        resp = api_post("/auth/register",
                                        json={"username": r_user,
                                              "email": r_email,
                                              "password": r_pass},
                                        auth=False)
                    if resp is None:
                        st.error("Cannot reach the server — is the backend running?")
                    elif resp.status_code == 201:
                        # Auto-login immediately
                        with st.spinner("Signing you in…"):
                            lr = api_post("/auth/login",
                                          json={"username": r_user, "password": r_pass},
                                          auth=False)
                        if lr and lr.status_code == 200:
                            d = lr.json()
                            st.session_state.token      = d["access_token"]
                            st.session_state.username   = d["username"]
                            st.session_state.user_id    = d["user_id"]
                            st.session_state.messages   = []
                            st.session_state.session_id = str(uuid.uuid4())
                            _load_persisted_chats()
                            st.session_state.page       = "upload"
                            st.session_state.is_new_user = True
                            st.rerun()
                        else:
                            st.success("Account created! Please sign in.")
                            st.session_state.auth_mode = "login"
                            st.rerun()
                    else:
                        st.error(safe_json(resp, "Registration failed."))

            st.markdown('<div class="toggle-row">Already have an account?&nbsp;</div>',
                        unsafe_allow_html=True)
            if st.button("Sign in →", key="go_login"):
                st.session_state.auth_mode = "login"
                st.rerun()


# ── SIDEBAR ─────────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        uname = st.session_state.username or "user"
        initials = uname[:2].upper()
        st.markdown(f"""
        <div class="sidebar-brand">
            <div class="sidebar-brand-icon">✳️</div>
            RAG Assistant
        </div>
        """, unsafe_allow_html=True)

        # Primary nav buttons
        is_upload_page = st.session_state.page == "upload"
        is_chat_page   = st.session_state.page == "chat"

        if st.button("📄  Upload Documents", key="nav_upload", use_container_width=True,
                      disabled=is_upload_page):
            st.session_state.page = "upload"
            st.rerun()

        if st.button("＋  New Chat", key="nav_new_chat", use_container_width=True):
            st.session_state.page       = "chat"
            st.session_state.messages   = []
            st.session_state.session_id = str(uuid.uuid4())
            st.rerun()

        # ── Chat history ─────────────────────────────────────────────────────
        if st.session_state.chats:
            st.markdown('<div class="sidebar-section">CHATS</div>', unsafe_allow_html=True)

            current_sid = st.session_state.session_id

            for sid, chat in list(st.session_state.chats.items()):
                is_active = is_chat_page and sid == current_sid
                title = chat["title"]
                display = (title[:28] + "…") if len(title) > 28 else title

                cols = st.columns([7, 1], vertical_alignment="center")
                with cols[0]:
                    if st.button(
                        display,
                        key=f"chat_open_{sid}",
                        use_container_width=True,
                        disabled=is_active,
                    ):
                        st.session_state.page = "chat"
                        st.session_state.session_id = sid
                        msgs = chat.get("messages")
                        if msgs is None:
                            msgs = _load_session_messages(sid)
                            st.session_state.chats[sid]["messages"] = msgs
                        st.session_state.messages = msgs
                        st.rerun()
                with cols[1]:
                    if st.button("×", key=f"chat_delete_{sid}", use_container_width=False):
                        api_delete(f"/chat/sessions/{sid}")
                        was_active = is_chat_page and sid == current_sid
                        del st.session_state.chats[sid]
                        if was_active:
                            st.session_state.messages = []
                            st.session_state.session_id = str(uuid.uuid4())
                        st.rerun()

        st.divider()
        st.caption("Groq · ChromaDB · LangGraph · DuckDuckGo")

        # User + sign out
        st.markdown(f"""
        <div class="sidebar-user">
            <div class="sidebar-user-avatar">{initials}</div>
            <span style="flex:1;color:#adadad;">{uname}</span>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Sign out", use_container_width=True, key="signout"):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            _init_state()
            st.rerun()


# ── CHAT MESSAGES ───────────────────────────────────────────────────────────────
def _render_sources(sources: list, used_web: bool):
    """Render the sources expander below an assistant message."""
    if not sources:
        return
    icon  = "🌐" if used_web else "📚"
    label = "Web Sources" if used_web else "Document Sources"
    with st.expander(f"{icon} {label} ({len(sources)})", expanded=False):
        for src in sources:
            score = f" · score {src['score']:.2f}" if src.get("score") else ""
            url   = f"  [{src['url']}]({src['url']})" if src.get("url") else ""
            pg    = f" · p.{src['page']}" if src.get("page") else ""
            st.markdown(f"**{src['source']}**{pg}{score}{url}")
            if src.get("excerpt"):
                st.caption(src["excerpt"])


def render_messages():
    if not st.session_state.messages:
        uname = st.session_state.username or "there"
        st.markdown(f"""
        <div class="welcome-wrap">
            <div class="welcome-icon">✳️</div>
            <div class="welcome-title">Hi {uname}, how can I help?</div>
            <div class="welcome-sub">I can search your documents or browse the web to answer your questions.</div>
        </div>
        """, unsafe_allow_html=True)
        return

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                used_web = msg.get("used_web", False)
                latency  = msg.get("latency", 0)
                total_tokens = msg.get("total_tokens")
                if latency or total_tokens:
                    badge = "🌐 web" if used_web else "📚 docs"
                    metrics = [badge]
                    if latency:
                        metrics.append(f"{latency}ms")
                    if total_tokens:
                        metrics.append(f"{total_tokens} tok")
                    st.caption(" · ".join(metrics))
                if msg.get("sources"):
                    _render_sources(msg["sources"], used_web)


# ── UPLOAD DOCUMENTS PAGE ────────────────────────────────────────────────────────
def render_upload_page():
    if not st.session_state.chats_loaded:
        _load_persisted_chats()
    render_sidebar()

    st.markdown("""
    <div style="max-width:760px;margin:40px auto 0;padding:0 24px;">
        <div style="display:flex;align-items:center;gap:12px;margin-bottom:8px;">
            <span style="font-size:28px;">📄</span>
            <span style="font-size:24px;font-weight:600;color:#ececec;letter-spacing:-0.3px;">
                Upload Documents
            </span>
        </div>
        <div style="font-size:14px;color:#8a8a8a;margin-bottom:28px;">
            Upload your documents to build the knowledge base. Supported formats:
            PDF, DOCX, DOC, TXT, MD, XLSX, XLS, CSV.
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_center = st.columns([1, 3, 1])[1]
    with col_center:
        uploaded_files = st.file_uploader(
            "Drop files here or click to browse",
            type=["pdf", "docx", "doc", "txt", "md", "xlsx", "xls", "csv"],
            accept_multiple_files=True,
            key=f"doc_uploader_{st.session_state.upload_nonce}",
            on_change=_mark_upload_pending,
        )

        should_upload = bool(uploaded_files) and bool(st.session_state.get("upload_pending"))
        if should_upload:
            results = []
            progress = st.progress(0, text="Uploading...")
            for i, f in enumerate(uploaded_files):
                progress.progress(
                    (i) / len(uploaded_files),
                    text=f"Indexing {f.name}…",
                )
                resp = _api_post_with_retry(
                    "/documents/upload",
                    files={"file": (f.name, f.getvalue(), f.type)},
                    timeout=600.0,  # first-time multimodal ingestion can be slow
                )
                if resp is None:
                    err = (st.session_state.get("_last_api_error") or "").lower()
                    if "timed out" in err or "timeout" in err:
                        detail = "Upload timed out while processing. The backend may still be indexing; wait and refresh."
                    else:
                        detail = "Server unreachable"
                    results.append({"file": f.name, "status": "error", "detail": detail})
                elif resp.status_code == 200:
                    d = resp.json()
                    if d.get("warning"):
                        results.append({"file": d["file"], "status": "warning", "detail": d["warning"]})
                    elif d.get("chunks", 0) == 0:
                        results.append({"file": d["file"], "status": "warning", "detail": "No text extracted"})
                    else:
                        results.append({"file": d["file"], "status": "success", "detail": f"{d['chunks']} chunks indexed"})
                else:
                    results.append({"file": f.name, "status": "error", "detail": safe_json(resp, "Upload failed")})
            progress.progress(1.0, text="Done!")

            st.session_state._upload_results = results
            st.session_state.upload_pending = False
            st.session_state.upload_nonce += 1
            st.rerun()

        # Show results from last batch upload
        if hasattr(st.session_state, "_upload_results") and st.session_state._upload_results:
            st.markdown("#### Upload Results")
            for r in st.session_state._upload_results:
                if r["status"] == "success":
                    st.success(f"✅  **{r['file']}** — {r['detail']}")
                elif r["status"] == "warning":
                    st.warning(f"⚠️  **{r['file']}** — {r['detail']}")
                else:
                    st.error(f"❌  **{r['file']}** — {r['detail']}")
            if st.button("Clear results", key="clear_upload_results"):
                st.session_state._upload_results = []
                st.rerun()

        st.markdown("---")

        # Document table
        st.markdown("#### Indexed Documents")
        resp = api_get("/documents", timeout=20.0)
        if resp and resp.status_code == 200:
            sources = resp.json().get("sources", [])
            if sources:
                # Table header
                hdr = st.columns([0.5, 4, 2, 1.5])
                hdr[0].markdown("**#**")
                hdr[1].markdown("**Document**")
                hdr[2].markdown("**Status**")
                hdr[3].markdown("**Action**")

                st.markdown(
                    "<hr style='margin:4px 0 8px;border-color:rgba(255,255,255,0.08);'>",
                    unsafe_allow_html=True,
                )

                for i, src in enumerate(sources, 1):
                    row = st.columns([0.5, 4, 2, 1.5], vertical_alignment="center")
                    row[0].markdown(f"`{i}`")
                    row[1].markdown(f"📄 {src}")
                    row[2].markdown(
                        '<span style="color:#4ade80;font-size:13px;">● Indexed</span>',
                        unsafe_allow_html=True,
                    )
                    if row[3].button("🗑️ Delete", key=f"del_doc_{src}", use_container_width=True):
                        dr = api_delete(f"/documents/{src}")
                        if dr and dr.status_code == 200:
                            st.session_state._upload_results = []
                            st.rerun()
                        else:
                            st.error(f"Failed to delete {src}")
            else:
                st.markdown(
                    '<div style="text-align:center;padding:40px 0;color:#6a6a6a;">'
                    'No documents indexed yet. Upload files above to get started.'
                    '</div>',
                    unsafe_allow_html=True,
                )
        else:
            err = st.session_state.get("_last_api_error", "")
            if err:
                st.warning(f"Could not fetch documents — {err}")
            else:
                st.warning("Could not fetch documents — is the backend running?")

        # Quick action to go to chat
        st.markdown("")
        if st.button("➜  Start chatting with your documents", key="go_to_chat",
                      use_container_width=True):
            st.session_state.page = "chat"
            st.rerun()


# ── CHAT PAGE ───────────────────────────────────────────────────────────────────
def render_chat():
    if not st.session_state.chats_loaded:
        _load_persisted_chats()
    render_sidebar()

    # ── Messages (oldest → newest) in a Streamlit-native fixed-height
    # scroll container. The `height=612` is a sentinel value we override
    # to `calc(100vh - 200px)` via CSS so the container takes all the
    # vertical space above the input pill. All chat content scrolls
    # within this container — nothing can ever appear behind the pill.
    with st.container(height=612, border=False):
        render_messages()

    # ── Mic button — CSS-fixed next to the submit button ────────────────────
    voice_text = None
    try:
        from audio_recorder_streamlit import audio_recorder
        # Rotating key forces a fresh component after each successful use,
        # otherwise the recorder stays "locked" with the last recording's bytes.
        mic_key = f"mic_{st.session_state.mic_counter}"
        audio = audio_recorder(
            text="",
            recording_color="#ffffff",
            neutral_color="#ffffff",
            icon_size="lg",
            key=mic_key,
        )
        if audio and len(audio) > 1000:
            with st.spinner("Transcribing…"):
                resp = api_post(
                    "/voice/transcribe",
                    files={"audio": ("rec.wav", audio, "audio/wav")},
                )
            if resp and resp.status_code == 200:
                voice_text = resp.json().get("transcript", "") or None
            st.session_state.mic_counter += 1   # next render = fresh mic component
    except ImportError:
        pass

    if voice_text:
        st.info(f"🎤 *{voice_text}*")

    # ── Chat input — placed at top level so Streamlit fixes it to the bottom ─
    user_input = st.chat_input("Message RAG Assistant…")

    # ── JS: style messages + scroll + snap mic ───────────────────────────────
    st.html("""
    <script>
    (function() {
        var doc = window.parent.document;
        function setImp(el, prop, val) {
            if (el) el.style.setProperty(prop, val, 'important');
        }

        /* ── 1. Auto-scroll to bottom ── */
        function scrollDown() {
            var main = doc.querySelector('[data-testid="stMain"]');
            if (main) main.scrollTop = main.scrollHeight;
        }
        scrollDown();
        setTimeout(scrollDown, 500);

        /* ── 2. Directly restyle chat messages (Claude layout) ── */
        function styleMessages() {
            var msgs = doc.querySelectorAll('[data-testid="stChatMessage"]');
            msgs.forEach(function(msg) {
                var isUser = msg.querySelector('[data-testid="chatAvatarIcon-user"]');

                /* Base row setup */
                msg.style.cssText += [
                    'display:flex',
                    'align-items:flex-start',
                    'gap:12px',
                    'max-width:680px',
                    'margin:0 auto 28px',
                    'background:transparent',
                    'border:none',
                    'padding:0',
                ].join('!important;') + '!important';

                /* Find content div (sibling of avatar container) */
                var kids = Array.from(msg.children);
                var avatarContainer = kids.find(function(k) {
                    return k.querySelector('[data-testid^="chatAvatarIcon"]');
                });
                var contentDiv = kids.find(function(k) { return k !== avatarContainer; });

                if (isUser) {
                    /* Row reversed: bubble on right */
                    msg.style.flexDirection = 'row-reverse';
                    if (contentDiv) {
                        contentDiv.style.cssText += [
                            'background:#242424',
                            'border:1px solid rgba(255,255,255,0.07)',
                            'border-radius:20px 4px 20px 20px',
                            'padding:10px 16px',
                            'max-width:65%',
                            'color:#e8e8e8',
                            'font-size:15px',
                            'line-height:1.6',
                        ].join('!important;') + '!important';
                    }
                } else {
                    /* Normal row: text on left */
                    msg.style.flexDirection = 'row';
                    if (contentDiv) {
                        contentDiv.style.cssText += [
                            'background:transparent',
                            'border:none',
                            'padding:0',
                            'max-width:100%',
                            'color:#e0e0e0',
                            'font-size:15px',
                            'line-height:1.8',
                        ].join('!important;') + '!important';
                    }
                }
            });
        }

        styleMessages();
        setTimeout(styleMessages, 600);

        /* ── 3. Hard-fix sidebar chat history row alignment ── */
        function styleSidebarChatRows() {
            var sidebar = doc.querySelector('[data-testid="stSidebar"]');
            if (!sidebar) return;
            var navKeywords = ['Upload Documents', 'New Chat', 'Sign out'];

            /* Style title buttons (chat history items) */
            var allButtons = sidebar.querySelectorAll('.stButton > button');
            allButtons.forEach(function(btn) {
                if (btn.closest('[data-testid="stPopoverContent"]')) return;
                var label = (btn.innerText || '').trim().replace(/\s+/g, ' ');
                if (!label) return;
                var isDelete = (label === '×' || label === '✕' || label.toLowerCase() === 'x');
                var isNav = navKeywords.some(function(k) { return label.indexOf(k) !== -1; });
                if (isNav) return;

                if (isDelete) {
                    var deleteCol = btn.closest('[data-testid="column"]');
                    if (deleteCol) {
                        setImp(deleteCol, 'width', '32px');
                        setImp(deleteCol, 'min-width', '32px');
                        setImp(deleteCol, 'max-width', '32px');
                        setImp(deleteCol, 'flex', '0 0 32px');
                    }
                    setImp(btn, 'display', 'inline-flex');
                    setImp(btn, 'justify-content', 'center');
                    setImp(btn, 'align-items', 'center');
                    setImp(btn, 'width', '32px');
                    setImp(btn, 'min-width', '32px');
                    setImp(btn, 'max-width', '32px');
                    setImp(btn, 'height', '32px');
                    setImp(btn, 'min-height', '32px');
                    setImp(btn, 'max-height', '32px');
                    setImp(btn, 'padding', '0');
                    setImp(btn, 'text-align', 'center');
                    setImp(btn, 'font-size', '21px');
                    setImp(btn, 'line-height', '1');
                    setImp(btn, 'white-space', 'nowrap');
                    setImp(btn, 'overflow', 'hidden');
                    return;
                }

                var col = btn.closest('[data-testid="column"]');
                if (col) setImp(col, 'min-width', '0');

                setImp(btn, 'display', 'flex');
                setImp(btn, 'justify-content', 'flex-start');
                setImp(btn, 'align-items', 'center');
                setImp(btn, 'width', '100%');
                setImp(btn, 'height', '36px');
                setImp(btn, 'min-height', '36px');
                setImp(btn, 'padding', '0 10px');
                setImp(btn, 'text-align', 'left');
                setImp(btn, 'white-space', 'nowrap');
                setImp(btn, 'overflow', 'hidden');
                btn.querySelectorAll('p, span, div').forEach(function(el) {
                    setImp(el, 'margin', '0');
                    setImp(el, 'padding', '0');
                    setImp(el, 'display', 'block');
                    setImp(el, 'width', '100%');
                    setImp(el, 'text-align', 'left');
                    setImp(el, 'white-space', 'nowrap');
                    setImp(el, 'overflow', 'hidden');
                    setImp(el, 'text-overflow', 'ellipsis');
                });
            });

        }

        styleSidebarChatRows();
        setTimeout(styleSidebarChatRows, 150);
        setTimeout(styleSidebarChatRows, 500);

        var sidebarObserver = new MutationObserver(function() {
            styleSidebarChatRows();
        });
        var sidebarRoot = doc.querySelector('[data-testid="stSidebar"]');
        if (sidebarRoot) {
            sidebarObserver.observe(sidebarRoot, { childList: true, subtree: true });
        }

        /* ── 4. Find audio_recorder iframe and snap it next to submit ── */
        function findMicBlock() {
            var iframes = doc.querySelectorAll('iframe');
            for (var i = 0; i < iframes.length; i++) {
                var t = iframes[i].title || '';
                if (t.indexOf('audio_recorder') !== -1) {
                    var block = iframes[i].closest('[data-testid="element-container"]')
                             || iframes[i].closest('[data-testid="stIFrame"]')
                             || iframes[i].parentElement;
                    return { block: block, iframe: iframes[i] };
                }
            }
            return null;
        }

        function placeMic() {
            var submit = doc.querySelector('[data-testid="stChatInputSubmitButton"]');
            var found  = findMicBlock();
            if (!submit || !found) return false;

            var r      = submit.getBoundingClientRect();
            var bottom = window.parent.innerHeight - r.bottom + (r.height - 36) / 2;
            var left   = r.left - 42;

            found.block.classList.add('mic-positioned');
            found.block.style.cssText = [
                'position:fixed',
                'bottom:' + Math.round(bottom) + 'px',
                'left:'   + Math.round(left)   + 'px',
                'width:36px', 'height:36px',
                'margin:0', 'padding:0',
                'z-index:9999',
                'display:flex', 'align-items:center', 'justify-content:center',
                'overflow:visible', 'visibility:visible', 'pointer-events:auto',
            ].join('!important;') + '!important';
            found.iframe.style.cssText = 'width:36px!important;height:36px!important;border:none!important;';
            return true;
        }

        var attempts = 0;
        function tryPlace() {
            var micOk = placeMic();
            if (micOk || ++attempts > 30) return;
            setTimeout(tryPlace, 200);
        }
        tryPlace();
        window.parent.addEventListener('resize', function() {
            setTimeout(placeMic, 150);
            setTimeout(styleMessages, 100);
            setTimeout(styleSidebarChatRows, 100);
        });
    })();
    </script>
    """, unsafe_allow_javascript=True)

    # ── Handle input ─────────────────────────────────────────────────────────
    final_input = user_input or voice_text
    if not final_input:
        return

    st.session_state.messages.append({"role": "user", "content": final_input})
    _register_current_chat(final_input)

    with st.spinner("Thinking…"):
        resp = api_post("/chat", json={
            "message": final_input,
            "session_id": st.session_state.session_id,
            "include_long_term_memory": True,
        }, timeout=120.0)

    if resp is None:
        st.error("Server unreachable — is the backend running?")
        return
    if resp.status_code == 422:
        st.error(f"⚙️ {safe_json(resp, 'Configuration error — check your .env API keys.')}")
        return
    if resp.status_code != 200:
        st.error(f"Error {resp.status_code}: {safe_json(resp)}")
        return

    d        = resp.json()
    usage    = d.get("token_usage") or {}
    total_tokens = usage.get("total_tokens")
    st.session_state.messages.append({
        "role":     "assistant",
        "content":  d["answer"],
        "sources":  d.get("sources", []),
        "used_web": d.get("used_web_search", False),
        "latency":  d.get("latency_ms", 0),
        "token_usage": usage,
        "total_tokens": total_tokens,
    })
    st.rerun()


# ── ROUTER ──────────────────────────────────────────────────────────────────────
if st.session_state.token is None:
    render_auth()
elif st.session_state.page == "upload":
    render_upload_page()
else:
    render_chat()
