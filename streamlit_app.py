"""Toronto 311 City Services Assistant – Streamlit (standalone, no separate backend)."""
import re
import copy
import asyncio
import streamlit as st
from datetime import datetime

from backend.agent import agent
from backend.rag import rag


@st.cache_resource(show_spinner="Loading knowledge base…")
def _init_rag():
    rag.initialize_knowledge_base()


_init_rag()

st.set_page_config(
    page_title="Toronto 311 | City Services Assistant",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* --- resets --- */
* { box-sizing: border-box; }
.block-container { padding: 0 !important; max-width: 100% !important; }
header[data-testid="stHeader"] { background: rgba(0,0,0,0) !important;
    color: white !important; }
.stDeployButton, #MainMenu, footer { display: none !important; }
section[data-testid="stSidebar"] { top: 0 !important; }
div.stMarkdown p { margin-bottom: 0 !important; }

/* ============================================================
   SIDEBAR  – light sky-blue theme
   ============================================================ */
section[data-testid="stSidebar"] > div {
    background: #EFF6FF !important;
    padding: 0 !important;
}
.sb-top {
    background: #DBEAFE;
    padding: 1.1rem 1rem .9rem;
    border-bottom: 1px solid #BFDBFE;
}
.sb-top-label {
    color: #1D4ED8; font-size: .85rem; font-weight: 800;
    text-transform: uppercase; letter-spacing: 1.2px;
}
.sb-section {
    padding: .9rem 1rem;
    border-bottom: 1px solid #BFDBFE;
}
.sb-label {
    color: #3B82F6; font-size: .78rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: 1.3px; margin-bottom: .6rem;
}
.sb-stat-row { display: flex; gap: 8px; }
.sb-stat {
    flex: 1; background: white;
    border: 1px solid #BFDBFE;
    border-radius: 10px; padding: .6rem .4rem; text-align: center;
}
.sb-stat-v { color: #1E3A8A; font-size: 1.2rem; font-weight: 800; }
.sb-stat-l { color: #60A5FA; font-size: .72rem; text-transform: uppercase; letter-spacing: .5px; }

/* sidebar buttons */
div[data-testid="stSidebar"] .stButton > button {
    background: white !important;
    border: 1.5px solid #BFDBFE !important;
    color: #1E40AF !important;
    border-radius: 10px !important;
    font-size: .93rem !important;
    width: 100% !important;
    padding: .52rem .85rem !important;
    margin-bottom: 5px !important;
    text-align: left !important;
    transition: all .15s !important;
    font-weight: 500 !important;
}
div[data-testid="stSidebar"] .stButton > button:hover {
    background: #DBEAFE !important;
    border-color: #3B82F6 !important;
    color: #1D4ED8 !important;
}
/* sidebar link buttons */
div[data-testid="stSidebar"] .stLinkButton > a {
    background: white !important;
    border: 1.5px solid #BFDBFE !important;
    color: #1E40AF !important;
    border-radius: 10px !important;
    font-size: .93rem !important;
    width: 100% !important;
    display: flex !important;
    padding: .52rem .85rem !important;
    margin-bottom: 5px !important;
    text-decoration: none !important;
    font-weight: 500 !important;
    transition: all .15s !important;
}
div[data-testid="stSidebar"] .stLinkButton > a:hover {
    background: #DBEAFE !important;
    border-color: #3B82F6 !important;
    color: #1D4ED8 !important;
}
div[data-testid="stSidebar"] * { color: #1E3A8A; }

/* ============================================================
   TOP NAV
   ============================================================ */
.top-nav {
    background: linear-gradient(135deg, #0C1B5C 0%, #1D4ED8 100%);
    height: 60px; display: flex; align-items: center;
    justify-content: space-between; padding: 0 2rem;
    box-shadow: 0 2px 12px rgba(0,0,0,.25);
    position: sticky; top: 0; z-index: 1000;
}
.nav-left { display: flex; align-items: center; gap: 14px; }
.nav-logo {
    width: 38px; height: 38px; border-radius: 50%; background: white;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.2rem; font-weight: 900; color: #1D4ED8; flex-shrink: 0;
}
.nav-title { color: white; font-size: 1.1rem; font-weight: 700; }
.nav-sub   { color: rgba(255,255,255,.75); font-size: .8rem; margin-top: 1px; }
.nav-right { display: flex; align-items: center; gap: 10px; }
.nav-status {
    display: flex; align-items: center; gap: 7px;
    background: rgba(255,255,255,.15); border: 1px solid rgba(255,255,255,.3);
    border-radius: 20px; padding: 5px 14px; color: white; font-size: .83rem;
}
.pulse-dot {
    width: 8px; height: 8px; border-radius: 50%; background: #4ADE80;
    animation: pulse 2s ease-in-out infinite;
}
@keyframes pulse { 0%,100%{opacity:1}50%{opacity:.4} }
.nav-badge {
    background: white; color: #1D4ED8; border-radius: 20px;
    padding: 5px 14px; font-size: .78rem; font-weight: 800; letter-spacing: .5px;
}

/* ============================================================
   MAIN AREA
   ============================================================ */
.main-bg {
    background: #F8FAFC; min-height: 100vh;
    padding-bottom: 120px;
}

/* ── Welcome card ── */
.wc-outer {
    max-width: 700px;
    margin: 5vh auto 0;
    padding: 0 1.4rem;
    text-align: center;
}
.wc {
    background: linear-gradient(135deg, #0C1B5C 0%, #1D4ED8 100%);
    border-radius: 24px;
    padding: 2.8rem 2.8rem 2.4rem;
    box-shadow: 0 12px 40px rgba(29,78,216,.35);
    display: inline-block;
    width: 100%;
}
.wc-badge {
    display: inline-block; background: rgba(255,255,255,.18);
    color: white; border-radius: 20px; padding: 5px 16px;
    font-size: .88rem; font-weight: 800; letter-spacing: 1.2px;
    text-transform: uppercase; margin-bottom: .9rem;
}

/* ── Quick action cards ── */
.qa-outer {
    max-width: 700px;
    margin: 1.4rem auto 0;
    padding: 0 1.4rem;
}
div.action-cards .stButton > button {
    background: #111827 !important;
    border: none !important;
    color: white !important;
    border-radius: 16px !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    padding: 1.4rem 1rem !important;
    width: 100% !important;
    min-height: 88px !important;
    transition: color .18s, background .18s, transform .18s !important;
    line-height: 1.4 !important;
    letter-spacing: .2px !important;
}
div.action-cards .stButton > button:hover {
    color: #FCD34D !important;
    background: #0F172A !important;
    transform: translateY(-3px) !important;
    box-shadow: 0 6px 20px rgba(0,0,0,.25) !important;
}
.wc-title {
    color: white; font-size: 2rem; font-weight: 900;
    margin-bottom: .55rem; line-height: 1.2;
}
.wc-sub {
    color: rgba(255,255,255,.85); font-size: 1.08rem;
    line-height: 1.65; font-weight: 400;
}


/* ── Chat area ── */
.chat-outer {
    max-width: 820px; margin: 0 auto;
    padding: 1.4rem 1.4rem 100px;
}

/* intent badge */
.ibadge {
    display: inline-flex; align-items: center; gap: 4px;
    padding: 3px 11px; border-radius: 14px;
    font-size: .78rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: .6px; margin-bottom: 6px;
}
.ib-general    { background:#EEF2FF; color:#4338CA; }
.ib-hazard     { background:#FFF7ED; color:#C2410C; }
.ib-permit     { background:#F0FDF4; color:#15803D; }
.ib-collection { background:#F0F9FF; color:#0369A1; }
.ib-scope      { background:#F8FAFC; color:#64748B; }

/* bubbles */
.bubble {
    padding: 1rem 1.2rem; border-radius: 18px;
    font-size: 1rem; line-height: 1.7; word-break: break-word;
}
.bubble.user {
    background: #1E293B; color: #F1F5F9;
    border-bottom-right-radius: 4px;
}
.bubble.bot {
    background: white; color: #1E293B;
    border: 1px solid #E2E8F0;
    border-bottom-left-radius: 4px;
    box-shadow: 0 2px 12px rgba(0,0,0,.06);
}
.bubble.bot ul { margin: .5rem 0 .5rem 1.3rem; padding: 0; }
.bubble.bot li { margin-bottom: 4px; }
.bubble.bot strong { color: #0F172A; }
.bubble.bot code {
    background: #F1F5F9; border-radius: 4px;
    padding: 2px 6px; font-size: .92rem;
}
.msg-time { font-size: .77rem; color: #94A3B8; margin-top: 5px; }
.msg-time.right { text-align: right; }

/* action cards */
.acard {
    border-radius: 13px; padding: 1rem 1.2rem;
    margin-top: .75rem; border-left: 4px solid; font-size: .93rem;
}
.acard-hazard     { background: #FFF7ED; border-color: #F97316; }
.acard-permit     { background: #F0FDF4; border-color: #22C55E; }
.acard-collection { background: #F0F9FF; border-color: #0EA5E9; }
.acard-lbl {
    font-size: .74rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: 1px;
    opacity: .65; margin-bottom: .55rem;
}
.acard-row { display: flex; align-items: baseline; gap: 8px; margin-bottom: 5px; }
.acard-key { color: #475569; font-size: .85rem; min-width: 100px; }
.acard-val { color: #0F172A; font-weight: 600; font-size: .93rem; }
.ticket-chip {
    font-family: monospace; background: rgba(0,0,0,.07);
    border-radius: 5px; padding: 2px 8px; font-weight: 700;
}
.day-big { font-size: 1.5rem; font-weight: 800; color: #0369A1; margin: 3px 0; }
.dec-yes   { color: #15803D; font-weight: 800; }
.dec-no    { color: #0369A1; font-weight: 800; }
.dec-maybe { color: #B45309; font-weight: 800; }

/* citations */
.cite-card {
    background: #F8FAFC; border: 1px solid #E2E8F0;
    border-radius: 9px; padding: .75rem 1rem;
    margin-bottom: 7px; font-size: .88rem;
}
.cite-src { color: #C8102E; font-weight: 700; font-size: .76rem;
            text-transform: uppercase; letter-spacing: .5px; }
.cite-title { color: #1E293B; font-weight: 600; margin: 3px 0; font-size: .9rem; }
.cite-exc { color: #64748B; line-height: 1.55; }

/* chat input */
.stChatInput textarea {
    border-radius: 26px !important;
    border: 2px solid #BFDBFE !important;
    padding: .9rem 1.3rem !important;
    font-size: 1rem !important;
    background: white !important;
    transition: border-color .2s !important;
}
.stChatInput textarea:focus {
    border-color: #C8102E !important;
    box-shadow: 0 0 0 3px rgba(200,16,46,.1) !important;
}
[data-testid="stChatInput"] {
    background: #F8FAFC;
    border-top: 1px solid #E2E8F0;
    padding: .85rem 1.2rem;
}
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def fmt_time():
    return datetime.now().strftime("%H:%M")


# ── Inline markdown → HTML ────────────────────────────────────────────────────

def md_to_html(text: str) -> str:
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text, flags=re.DOTALL)
    text = re.sub(r'(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)', r'<em>\1</em>', text)
    text = re.sub(r'`(.+?)`', r'<code>\1</code>', text)
    lines = text.split("\n")
    out, in_list = [], False
    for raw in lines:
        line = raw.strip()
        if line.startswith("- ") or line.startswith("• "):
            if not in_list:
                out.append("<ul>"); in_list = True
            out.append(f"<li>{line[2:]}</li>")
        elif line in ("", "---"):
            if in_list:
                out.append("</ul>"); in_list = False
            out.append("<br>")
        else:
            if in_list:
                out.append("</ul>"); in_list = False
            out.append(f"<span>{line}</span><br>")
    if in_list:
        out.append("</ul>")
    return "".join(out)


# ── Intent badge ──────────────────────────────────────────────────────────────

_BADGE = {
    "general_inquiry":   ("ib-general",    "📖 Bylaw Info"),
    "hazard_report":     ("ib-hazard",     "🚧 Hazard Report"),
    "permit_screener":   ("ib-permit",     "🏗 Permit Check"),
    "collection_lookup": ("ib-collection", "♻️ Waste Collection"),
    "out_of_scope":      ("ib-scope",      "↩ Out of Scope"),
}


def intent_badge_html(intent: str) -> str:
    cls, label = _BADGE.get(intent, ("ib-scope", intent))
    return f'<span class="ibadge {cls}">{label}</span>'


# ── Action card ───────────────────────────────────────────────────────────────

def action_card_html(action: dict) -> str:
    if not action:
        return ""
    t, status, data = action.get("type", ""), action.get("status", ""), action.get("data", {})

    if t == "hazard_report":
        if status == "completed":
            tid = data.get("ticket_id", "—")
            return f"""<div class="acard acard-hazard">
<div class="acard-lbl">🚧 Hazard Report Submitted</div>
<div class="acard-row"><span class="acard-key">Ticket ID</span>
<span class="ticket-chip">{tid}</span></div>
<div class="acard-row"><span class="acard-key">Location</span>
<span class="acard-val">{data.get('location','—')}</span></div>
<div class="acard-row"><span class="acard-key">Hazard Type</span>
<span class="acard-val">{data.get('hazard_type','—')}</span></div>
<div class="acard-row"><span class="acard-key">Response Time</span>
<span class="acard-val">24–48 hours</span></div>
</div>"""
        n = sum(1 for f in ["location", "hazard_type", "description"] if f in data)
        rows = "".join(
            f'<div class="acard-row"><span class="acard-key">'
            f'{k.replace("_"," ").title()}</span>'
            f'<span class="acard-val">{data[k]}</span></div>'
            for k in ["location", "hazard_type", "description"] if k in data
        )
        return f"""<div class="acard acard-hazard">
<div class="acard-lbl">🚧 Report in Progress — Step {n+1}/3</div>
{rows}</div>"""

    if t == "permit_screener":
        raw = data.get("decision", "")
        u = raw.upper()
        cls = "dec-yes" if "YES" in u else "dec-no" if "NO" in u else "dec-maybe"
        icon = "✅" if "YES" in u else "🔵" if "NO" in u else "⚠️"
        proj = data.get("project_description", "")[:70]
        return f"""<div class="acard acard-permit">
<div class="acard-lbl">🏗 Permit Screening Result</div>
<div class="acard-row"><span class="acard-key">Decision</span>
<span class="{cls}">{icon} {raw or 'See response'}</span></div>
<div class="acard-row"><span class="acard-key">Project</span>
<span class="acard-val">{proj}…</span></div>
</div>"""

    if t == "collection_lookup":
        day = data.get("collection_day", "—")
        pc = data.get("postal_code", "")
        pc_fmt = f"{pc[:3]} {pc[3:]}" if len(pc) >= 6 else pc
        return f"""<div class="acard acard-collection">
<div class="acard-lbl">♻️ Collection Schedule</div>
<div class="acard-row"><span class="acard-key">Postal Code</span>
<span class="acard-val">{pc_fmt}</span></div>
<div class="day-big">{day}</div>
<div style="font-size:.83rem;color:#0369A1">Weekly collection day</div>
</div>"""

    return ""


# ── Message renderer ──────────────────────────────────────────────────────────

def render_message(msg: dict):
    role      = msg["role"]
    content   = msg["content"]
    timestamp = msg.get("timestamp", "")
    intent    = msg.get("intent", "")
    action    = msg.get("action")
    citations = msg.get("citations", [])

    if role == "user":
        _, col = st.columns([0.22, 0.78])
        with col:
            st.markdown(
                f'<div class="bubble user">{content}</div>'
                f'<div class="msg-time right">{timestamp}</div>',
                unsafe_allow_html=True,
            )
        return

    col, _ = st.columns([0.88, 0.12])
    with col:
        if intent:
            st.markdown(intent_badge_html(intent), unsafe_allow_html=True)
        st.markdown(
            f'<div class="bubble bot">{md_to_html(content)}</div>',
            unsafe_allow_html=True,
        )
        acard = action_card_html(action)
        if acard:
            st.markdown(acard, unsafe_allow_html=True)
        if citations:
            with st.expander(f"📎 {len(citations)} source{'s' if len(citations)>1 else ''}"):
                for c in citations:
                    st.markdown(f"""<div class="cite-card">
<div class="cite-src">{c.get('source','')}</div>
<div class="cite-title">{c.get('title','')}</div>
<div class="cite-exc">{c.get('excerpt','')[:200]}…</div>
</div>""", unsafe_allow_html=True)
        st.markdown(f'<div class="msg-time">{timestamp}</div>', unsafe_allow_html=True)

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────────
for k, v in [("messages", []), ("conv_state", {}), ("quick_send", None)]:
    if k not in st.session_state:
        st.session_state[k] = v


# ── Top nav ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="top-nav">
  <div class="nav-left">
    <div class="nav-logo">T</div>
    <div>
      <div class="nav-title">Toronto 311 · City Services</div>
      <div class="nav-sub">Municipal Bylaw &amp; Services Assistant</div>
    </div>
  </div>
  <div class="nav-right">
    <div class="nav-status"><span class="pulse-dot"></span>System Online</div>
    <div class="nav-badge">AI ASSISTANT</div>
  </div>
</div>""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    n_msg  = len(st.session_state.messages)
    n_user = sum(1 for m in st.session_state.messages if m["role"] == "user")

    st.markdown(f"""
<div class="sb-top">
  <div class="sb-top-label">Session Overview</div>
</div>
<div class="sb-section">
  <div class="sb-label">Statistics</div>
  <div class="sb-stat-row">
    <div class="sb-stat">
      <div class="sb-stat-v">{n_msg}</div>
      <div class="sb-stat-l">Messages</div>
    </div>
    <div class="sb-stat">
      <div class="sb-stat-v">{n_user}</div>
      <div class="sb-stat-l">Queries</div>
    </div>
    <div class="sb-stat">
      <div class="sb-stat-v">🟢</div>
      <div class="sb-stat-l">Status</div>
    </div>
  </div>
</div>""", unsafe_allow_html=True)

    st.markdown('<div class="sb-section"><div class="sb-label">Conversation</div>',
                unsafe_allow_html=True)
    if st.button("➕  New Conversation", key="new_conv", use_container_width=True):
        st.session_state.messages = []
        st.session_state.conv_state = {}
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sb-section"><div class="sb-label">City Services</div>',
                unsafe_allow_html=True)
    st.link_button("📞  Toronto 311 — Call or Chat",
                   "https://www.toronto.ca/home/311-toronto-at-your-service/",
                   use_container_width=True)
    st.link_button("🏗  Toronto Building Permits",
                   "https://www.toronto.ca/city-government/planning-development/",
                   use_container_width=True)
    st.link_button("♻️  Waste & Recycling",
                   "https://www.toronto.ca/services-payments/recycling-organics-garbage/",
                   use_container_width=True)
    st.link_button("🌳  Urban Forestry & Trees",
                   "https://www.toronto.ca/services-payments/water-environment/trees/",
                   use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("""
<div style="padding:.85rem 1rem;color:#3B82F6;font-size:.77rem;line-height:1.7">
  Toronto Bylaw Agent v1.0<br>
  <span style="color:#93C5FD">RSM8430 · University of Toronto</span><br>
  <span style="color:#93C5FD">Source: Toronto Municipal Code</span>
</div>""", unsafe_allow_html=True)


# ── Welcome card (only when no messages) ─────────────────────────────────────
if not st.session_state.messages:
    st.markdown("""
<div class="wc-outer">
  <div class="wc">
    <div class="wc-badge">Toronto 311 · AI Assistant</div>
    <div class="wc-title">How can I help you today?</div>
    <div class="wc-sub">
      I'm your Toronto municipal services assistant. Ask me about bylaws,
      report a hazard, check permit requirements, or look up your waste collection schedule.
    </div>
  </div>
</div>""", unsafe_allow_html=True)

    st.markdown('<div class="qa-outer"><div class="action-cards">',
                unsafe_allow_html=True)
    qa_cols = st.columns(3)
    qa_items = [
        ("🚧", "Report a Hazard",
         "I need to report a hazard in Toronto"),
        ("🏗", "Check Permit Required",
         "Do I need a building permit for my project?"),
        ("♻️", "Waste Collection Day",
         "When is my garbage collection day? My postal code is M5V 3A8"),
    ]
    for i, (icon, label, prompt) in enumerate(qa_items):
        with qa_cols[i]:
            if st.button(f"{icon}\n{label}", key=f"qa_card_{i}",
                         use_container_width=True):
                st.session_state.quick_send = prompt
    st.markdown('</div></div>', unsafe_allow_html=True)

# ── Chat messages ─────────────────────────────────────────────────────────────
if st.session_state.messages:
    st.markdown('<div class="chat-outer">', unsafe_allow_html=True)
    for msg in st.session_state.messages:
        render_message(msg)
    st.markdown('</div>', unsafe_allow_html=True)


# ── Chat input ────────────────────────────────────────────────────────────────
prompt = st.session_state.pop("quick_send", None)
chat_val = st.chat_input(
    "Ask me about Toronto bylaws, report hazards, check permits, or look up waste collection…"
)
if chat_val:
    prompt = chat_val

if prompt:
    st.session_state.messages.append({
        "role": "user", "content": prompt, "timestamp": fmt_time(),
    })

    with st.spinner(""):
        try:
            resp_obj = asyncio.run(
                agent.process_message(prompt, copy.deepcopy(st.session_state.conv_state))
            )
            # Update multi-turn conversation state from response
            if resp_obj.action:
                if resp_obj.action.get("status") == "in_progress":
                    st.session_state.conv_state = resp_obj.action["data"]
                elif resp_obj.action.get("status") == "completed":
                    st.session_state.conv_state = {}

            resp = resp_obj.to_dict()
            action = resp.get("action")
            if action and action.get("type") == "permit_screener":
                msg_text = resp.get("message", "")
                for kw in ["YES", "NO", "POSSIBLY"]:
                    if kw in msg_text.upper():
                        action.setdefault("data", {})["decision"] = kw
                        break
            st.session_state.messages.append({
                "role":      "assistant",
                "content":   resp.get("message", ""),
                "intent":    resp.get("intent", ""),
                "action":    action,
                "citations": resp.get("citations", []),
                "timestamp": fmt_time(),
            })
        except Exception as e:
            st.session_state.messages.append({
                "role":      "assistant",
                "content":   f"⚠️ An error occurred: {e}",
                "timestamp": fmt_time(),
            })

    st.rerun()
