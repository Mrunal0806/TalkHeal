import streamlit as st
from components.reset_page import show_reset_password_page

# --- Grab token from URL ---
reset_token = st.query_params.get("token")

# --- Initialize session state ---
if "reset_token" not in st.session_state:
    st.session_state.reset_token = None
if "show_reset_page" not in st.session_state:
    st.session_state.show_reset_page = False

# --- If token exists in URL, save it and show reset page ---
if reset_token:
    st.session_state.reset_token = reset_token
    st.session_state.show_reset_page = True

def run():
    if st.session_state.show_reset_page and st.session_state.reset_token:
        show_reset_password_page()
    else:
        st.info("No reset token provided. Please check your email link.")

if __name__ == "__main__":
    run()
