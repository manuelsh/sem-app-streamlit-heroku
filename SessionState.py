"""Compatibility wrapper around Streamlit's public session-state API."""

import streamlit as st


def get(**kwargs):
    """Initialize missing keys and return the current session state."""
    for key, value in kwargs.items():
        if key not in st.session_state:
            st.session_state[key] = value
    return st.session_state
