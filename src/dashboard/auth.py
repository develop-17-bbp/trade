"""
Dashboard Authentication
========================
Simple session-based authentication for the Streamlit dashboard.
Set DASHBOARD_PASS environment variable to enable.
If not set, dashboard is accessible without login (development mode).
"""

import os
import hashlib
import hmac
import streamlit as st


def check_auth() -> bool:
    """
    Verify dashboard authentication.
    Returns True if authenticated or if auth is disabled (no DASHBOARD_PASS set).
    """
    password = os.environ.get("DASHBOARD_PASS", "")
    if not password:
        # No password configured — show dev-mode warning in sidebar
        return True

    # Check if already authenticated this session
    if st.session_state.get("authenticated"):
        return True

    # Show login form
    st.markdown("""
    <div style="display:flex; justify-content:center; align-items:center; min-height:60vh;">
        <div style="background: rgba(15,15,30,0.9); padding: 40px; border-radius: 16px;
                    border: 1px solid rgba(0,234,255,0.2); max-width: 400px; width: 100%;
                    box-shadow: 0 0 40px rgba(0,234,255,0.05);">
            <div style="text-align:center; margin-bottom:24px;">
                <div style="font-size:2rem;">&#x1F512;</div>
                <div style="font-family:monospace; font-size:1.2rem; color:#00eaff; letter-spacing:2px;">
                    TRADING DESK
                </div>
                <div style="font-size:0.7rem; color:#555; margin-top:4px;">AUTHENTICATION REQUIRED</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.form("login_form"):
        entered = st.text_input("Password", type="password", placeholder="Enter dashboard password")
        submitted = st.form_submit_button("Authenticate", type="primary")

        if submitted:
            # Constant-time comparison to prevent timing attacks
            if hmac.compare_digest(
                hashlib.sha256(entered.encode()).hexdigest(),
                hashlib.sha256(password.encode()).hexdigest()
            ):
                st.session_state["authenticated"] = True
                st.rerun()
            else:
                st.error("Invalid credentials")

    return False


def show_dev_mode_warning():
    """Show a warning if auth is disabled."""
    if not os.environ.get("DASHBOARD_PASS"):
        st.sidebar.warning(
            "**Auth disabled** — Set `DASHBOARD_PASS` env var to secure this dashboard.",
            icon="\u26a0\ufe0f"
        )
