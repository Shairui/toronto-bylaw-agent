"""Streamlit frontend for Toronto Bylaw Agent."""
import streamlit as st
import requests
import json
from datetime import datetime
from typing import Optional

# Configuration
API_URL = "http://localhost:8000"

# Page config
st.set_page_config(
    page_title="Toronto Bylaw Agent",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        max-width: 900px;
        margin: 0 auto;
    }
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .citation {
        background-color: #f0f2f6;
        padding: 0.5rem;
        border-left: 3px solid #0066cc;
        margin-top: 0.5rem;
        font-size: 0.85rem;
    }
    .action-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-top: 1rem;
        border-left: 4px solid #00a8e8;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "conversation_state" not in st.session_state:
    st.session_state.conversation_state = {}


def create_conversation():
    """Create a new conversation."""
    try:
        response = requests.post(
            f"{API_URL}/conversations",
            json={"title": "New Conversation"}
        )
        if response.status_code == 200:
            data = response.json()
            st.session_state.conversation_id = data["id"]
            st.session_state.messages = []
            st.session_state.conversation_state = {}
            return data["id"]
    except Exception as e:
        st.error(f"Error creating conversation: {e}")
    return None


def send_message(user_message: str):
    """Send a message to the agent."""
    if not st.session_state.conversation_id:
        create_conversation()
    
    try:
        response = requests.post(
            f"{API_URL}/conversations/{st.session_state.conversation_id}/messages",
            json={"conversation_id": st.session_state.conversation_id, "message": user_message}
        )
        
        if response.status_code == 200:
            agent_response = response.json()
            
            # Add messages to session
            st.session_state.messages.append({
                "role": "user",
                "content": user_message,
                "timestamp": datetime.now()
            })
            
            st.session_state.messages.append({
                "role": "assistant",
                "content": agent_response["message"],
                "intent": agent_response["intent"],
                "citations": agent_response.get("citations", []),
                "action": agent_response.get("action"),
                "timestamp": datetime.now()
            })
            
            # Update conversation state if needed
            if agent_response.get("action") and agent_response["action"]["type"] == "hazard_report":
                st.session_state.conversation_state = agent_response["action"]["data"]
            
            return True
        else:
            st.error(f"Error: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        st.error(f"Error sending message: {e}")
        return False


# Header
st.markdown("# 🏛️ Toronto Bylaw Agent")
st.markdown("Your intelligent assistant for Toronto municipal services and bylaws")

# Sidebar
with st.sidebar:
    st.markdown("## 📋 About This Agent")
    st.markdown("""
    This AI agent helps you with:
    - **Bylaw Guidance**: Questions about Toronto regulations
    - **Hazard Reporting**: Report potholes, fallen trees, debris
    - **Permit Screening**: Determine if you need a building permit
    - **Waste Schedule**: Look up your collection days
    """)
    
    st.markdown("---")
    
    if st.button("🔄 New Conversation", use_container_width=True):
        create_conversation()
        st.rerun()
    
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.conversation_state = {}
        st.rerun()
    
    st.markdown("---")
    st.markdown("### 📞 Quick Links")
    st.markdown("""
    - [Toronto 311](https://www.toronto.ca/311)
    - [Building Permits](https://www.toronto.ca/building-permits)
    - [Waste Collection](https://www.toronto.ca/waste)
    - [Municipal Code](https://www.toronto.ca/municipal-code)
    """)


# Main chat interface
st.markdown("---")

# Display chat history
for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.write(msg["content"])
    else:
        with st.chat_message("assistant"):
            st.write(msg["content"])
            
            # Display citations if present
            if msg.get("citations"):
                st.markdown("### 📚 Sources")
                for citation in msg["citations"]:
                    with st.expander(f"📄 {citation.get('title', 'Source')}"):
                        st.markdown(f"**Source**: {citation.get('source', 'Unknown')}")
                        st.markdown(f"**Excerpt**: {citation.get('excerpt', 'N/A')}")
            
            # Display action info if present
            if msg.get("action"):
                action = msg["action"]
                if action["type"] == "hazard_report" and action["status"] == "completed":
                    st.markdown(f"""
                    <div class="action-box">
                    <strong>✅ Hazard Report Submitted</strong><br>
                    Ticket ID: {action['data'].get('ticket_id', 'N/A')}<br>
                    Location: {action['data'].get('location', 'N/A')}<br>
                    Type: {action['data'].get('hazard_type', 'N/A')}
                    </div>
                    """, unsafe_allow_html=True)
                elif action["type"] == "permit_screener":
                    st.markdown(f"""
                    <div class="action-box">
                    <strong>🏗️ Permit Screening Result</strong><br>
                    Project: {action['data'].get('project_description', 'N/A')[:100]}...
                    </div>
                    """, unsafe_allow_html=True)
                elif action["type"] == "collection_lookup":
                    st.markdown(f"""
                    <div class="action-box">
                    <strong>♻️ Collection Schedule</strong><br>
                    Postal Code: {action['data'].get('postal_code', 'N/A')}<br>
                    Collection Day: {action['data'].get('collection_day', 'N/A')}
                    </div>
                    """, unsafe_allow_html=True)


# Input area
st.markdown("---")
col1, col2 = st.columns([1, 0.15])

with col1:
    user_input = st.chat_input("Ask me about Toronto bylaws, report hazards, check permits, or look up waste collection...")

with col2:
    if st.button("Send", use_container_width=True, key="send_btn"):
        if user_input:
            send_message(user_input)
            st.rerun()

# Handle Enter key
if user_input:
    send_message(user_input)
    st.rerun()


# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.85rem;">
    <p>Toronto Bylaw Agent | Powered by LLM | For questions, contact Toronto 311 at 416-392-8111</p>
</div>
""", unsafe_allow_html=True)
