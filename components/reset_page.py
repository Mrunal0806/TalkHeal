import streamlit as st
import time
from auth.auth_utils import reset_password, check_user , verify_token_count
from auth.mail_utils import send_reset_email
from components.login_page import show_login_page
from auth.jwt_utils import verify_reset_token


def show_reset_password_page():
    """Renders the password reset page with the modern dark theme."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Baloo+2:wght@600&display=swap');
        @keyframes floatHearts {
            0% { transform: translateY(0) scale(1); opacity: 1; }
            100% { transform: translateY(-120px) scale(1.3); opacity: 0; }
        }
        body, html {
            height: 100%;
            min-height: 100vh;
            background: linear-gradient(135deg, #ffe0f0 0%, #ffd6e0 100%);
            font-family: 'Baloo 2', cursive;
        }
        [data-testid="stSidebar"] { display: none; }
        [data-testid="stHeader"] { display: none; }
        .block-container {
            background: linear-gradient(135deg, #fff0f6 60%, #ffe0f0 100%);
            border-radius: 32px;
            max-width: 420px;
            margin: auto;
            margin-top: 60px;
            padding: 2.7rem 3rem 2.2rem 3rem;
            border: 2.5px solid #ffb6d5;
            box-shadow: 0 0 32px 8px #ffd6e0, 0 10px 40px rgba(255, 182, 213, 0.35);
            animation: fadeIn 0.7s ease-out;
            transition: box-shadow 0.2s;
        }
        .block-container:hover {
            box-shadow: 0 0 48px 16px #ffb6d5, 0 10px 40px rgba(255, 182, 213, 0.45);
        }
        .logo-animated {
            display: flex;
            justify-content: center;
            align-items: center;
            margin-bottom: 0.7rem;
        }
        .logo-animated img {
            width: 74px;
            height: 74px;
            border-radius: 50%;
            box-shadow: 0 0 32px 8px #ffb6d5, 0 2px 16px #ffd6e0;
            background: radial-gradient(circle at 60% 40%, #ffe0f0 70%, #ffb6d5 100%);
            padding: 8px;
            animation: floatLogo 2s infinite alternate ease-in-out;
        }
        @keyframes floatLogo {
            0% { transform: translateY(0); }
            100% { transform: translateY(-12px); }
        }
        .auth-title {
            text-align: center;
            font-size: 2.5rem;
            font-weight: 800;
            color: #ff69b4;
            margin-bottom: 0.5rem;
            font-family: 'Baloo 2', cursive;
            letter-spacing: 1px;
        }
        .subtitle {
            text-align: center;
            font-size: 1.1rem;
            color: #ffb6d5;
            margin-bottom: 2.5rem;
            font-family: 'Baloo 2', cursive;
        }
        .auth-input input {
            width: 100%;
            padding: 0.8rem 1rem;
            margin-bottom: 1.25rem;
            border-radius: 12px;
            border: 1.5px solid #ffb6d5;
            background-color: #fff6fa;
            color: #ff69b4;
            font-size: 1rem;
            font-family: 'Baloo 2', cursive;
            transition: all 0.2s ease-in-out;
        }
        .auth-input input::placeholder {
            color: #ffb6d5;
        }
        .auth-input input:focus {
            outline: none;
            border-color: #ff69b4;
            background-color: #ffe0f0;
            box-shadow: 0 0 0 3px rgba(255, 182, 213, 0.3);
        }
        .auth-button {
            display: flex;
            justify-content: center;
            width: 100%;
        }
        .auth-button button {
            width: 100%;
            padding: 0.95rem;
            border-radius: 18px;
            font-weight: 700;
            font-size: 1.18rem;
            border: none;
            color: #fff0f6;
            margin-top: 0.5rem;
            cursor: pointer;
            background: linear-gradient(90deg, #ffb6d5 0%, #ff69b4 100%);
            box-shadow: 0 2px 12px rgba(255, 182, 213, 0.22);
            transition: background 0.2s, box-shadow 0.2s;
            position: relative;
        }
        .auth-button button::after {
            content: \" 💖\";
            font-size: 1.1rem;
            margin-left: 6px;
        }
        .auth-button button:hover {
            background: linear-gradient(90deg, #ff69b4 0%, #ffb6d5 100%);
            box-shadow: 0 4px 24px rgba(255, 182, 213, 0.32);
        }
        .switch-link {
            display: flex;
            justify-content: center;
            width: 100%;
            margin-top: 1.5rem;
        }
        .switch-link button {
            background: none;
            color: #ff69b4;
            border: none;
            font-size: 1rem;
            text-decoration: none;
            cursor: pointer;
            font-family: 'Baloo 2', cursive;
            transition: color 0.2s;
        }
        .switch-link button:hover {
            color: #ffb6d5;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    title = "Reset Your Password"
    st.markdown('<div class="logo-animated"><img src="https://raw.githubusercontent.com/eccentriccoder01/TalkHeal/main/static_files/TalkHealLogo.png" alt="Logo"/></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="auth-title" style="color:#ffb6d5;">{title}</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Update Your password 🩷</div>', unsafe_allow_html=True)

    # Retrieve reset token from session state
    reset_token = st.session_state.get("reset_token", None)
    if not reset_token:
        st.error("Invalid or missing reset token. Please request a new password reset link.")
        return

    form_container = st.container()
    is_verified , data=verify_reset_token(reset_token)
    if not is_verified or not data:
        st.error("Your reset link is invalid or has expired. Please request a new one.")
        st.session_state.show_reset_page = False
        st.session_state.is_signup = False
        st.session_state.reset_token = None
        st.stop()

    # Check token vs DB (updated_at still matches)
    valid, msg = verify_token_count(data.get("email"), data.get("pwd_update"))
    if not valid:
        st.error("Link Already Used .Please request a new one.")
        st.info("Redirecting to the forget page in 3 seconds...")
        st.session_state.show_reset_page = False
        st.session_state.show_forget_page = True
        st.session_state.reset_token = None
        time.sleep(3)
        st.switch_page("TalkHeal.py")
        
    with form_container:
        new_password = st.text_input(
            "New Password", type="password", placeholder="••••••••",
            label_visibility="collapsed", key="new_password"
        )
        confirm_password = st.text_input(
            "Confirm Password", type="password", placeholder="••••••••",
            label_visibility="collapsed", key="confirm_password"
        )

        st.markdown('<div class="auth-button">', unsafe_allow_html=True)
        if st.button("Reset Password", key="reset_password_submit"):
            if not new_password or not confirm_password:
                st.warning("Please fill out all fields.")
            elif new_password != confirm_password:
                st.error("Passwords do not match.")
            elif len(new_password) < 8:
                st.error("Password must be at least 8 characters long.")
            else:
                success, message = reset_password(data.get("email"), new_password)
                if success:
                    st.success("Password reset successfully! You can now log in.")
                    st.info("Redirecting to the Login page in 3 seconds...")
                    st.session_state.show_reset_page = False
                    st.session_state.reset_token = None
                    st.session_state.show_login_page = True
                    time.sleep(3)
                    st.switch_page("TalkHeal.py")
                else:
                    st.error(message)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="auth-button">', unsafe_allow_html=True)
        if st.button("Back to Login", key="switch_to_login"):
            st.session_state.show_reset_page = False
            st.session_state.show_login_page = True
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

# Main app logic in resetpage.py
if __name__ == "__main__":
    if "show_reset_page" not in st.session_state:
        st.session_state.show_reset_page = False
    if "show_login_page" not in st.session_state:
        st.session_state.show_login_page = True
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    # ✅ Only set reset page if token exists *and* session doesn't already have a result
    if not st.session_state.get("reset_token"):
        reset_token = st.query_params.get("token")
        if reset_token:
            st.session_state.reset_token = reset_token
            st.session_state.show_reset_page = True
            st.session_state.show_login_page = False

    # 🚪 Routing logic
    if not st.session_state.authenticated:
        if st.session_state.show_reset_page:
            show_reset_password_page()
        else:
            show_login_page()
    else:
        st.title(f"🎉 Welcome, {st.session_state.user_name}!")
        st.success("You're logged in!")
        if st.button("Logout"):
            st.session_state.authenticated = False
            st.session_state.user_name = ""
            st.session_state.show_reset_page = False
            st.session_state.show_login_page = True
            # 🚨 clear reset_token if any
            st.session_state.reset_token = None
            st.rerun()
