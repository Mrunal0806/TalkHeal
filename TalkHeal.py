import streamlit as st
import google.generativeai as genai
import cv2

# --- LOCAL IMPORTS ---
from auth.auth_utils import init_db
from components.login_page import show_login_page
from hand_gesture_recognition_mediapipe.inference import gesture_mode
from core.utils import save_conversations, load_conversations, get_current_time, create_new_conversation
from core.config import configure_gemini
from css.styles import apply_custom_css
from components.header import render_header
from components.sidebar import render_sidebar
from components.chat_interface import render_chat_interface, handle_chat_input, render_session_controls
from components.mood_dashboard import render_mood_dashboard
from components.emergency_page import render_emergency_page
from components.focus_session import render_focus_session
from components.profile import apply_global_font_size

# --- PAGE CONFIG ---
st.set_page_config(page_title="TalkHeal", page_icon="💬", layout="wide")

# --- HIDE SIDEBAR NAVIGATION ---
st.markdown(
    "<style>div[data-testid='stSidebarNav'] {display: none;}</style>",
    unsafe_allow_html=True
)

# --- DB INIT ---
if "db_initialized" not in st.session_state:
    init_db()
    st.session_state["db_initialized"] = True

# --- AUTH STATE ---
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "show_signup" not in st.session_state:
    st.session_state.show_signup = False

# --- LOGIN PAGE ---
if not st.session_state.authenticated:
    show_login_page()
    st.stop()

# --- TOP RIGHT BUTTONS ---
if st.session_state.get("authenticated", False):
    col_spacer, col_theme, col_emergency, col_about, col_logout = st.columns([0.7, 0.1, 0.35, 0.2, 0.2])

    with col_spacer:
        pass
    with col_theme:
        is_dark = st.session_state.get('dark_mode', False)
        if st.button("🌙" if is_dark else "☀️", key="top_theme_toggle", use_container_width=True):
            st.session_state.dark_mode = not is_dark
            st.session_state.theme_changed = True
            st.rerun()
    with col_emergency:
        if st.button("🚨 Emergency Help", key="emergency_main_btn", use_container_width=True, type="secondary"):
            st.session_state.show_emergency_page = True
            st.rerun()
    with col_about:
        if st.button("ℹ️ About", key="about_btn", use_container_width=True):
            st.switch_page("pages/About.py")
    with col_logout:
        if st.button("Logout", key="logout_btn", use_container_width=True):
            for key in ["authenticated", "user_email", "user_name", "show_signup"]:
                st.session_state.pop(key, None)
            st.rerun()

# --- SESSION STATE INIT ---
defaults = {
    "chat_history": [],
    "conversations": load_conversations(),
    "active_conversation": -1,
    "show_emergency_page": False,
    "show_focus_session": False,
    "show_mood_dashboard": False,
    "sidebar_state": "expanded",
    "mental_disorders": [
        "Depression & Mood Disorders", "Anxiety & Panic Disorders", "Bipolar Disorder",
        "PTSD & Trauma", "OCD & Related Disorders", "Eating Disorders",
        "Substance Use Disorders", "ADHD & Neurodevelopmental", "Personality Disorders",
        "Sleep Disorders"
    ],
    "selected_tone": "Compassionate Listener",
    "gesture_mode": False
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# --- APPLY CONFIG & STYLES ---
apply_global_font_size()
apply_custom_css()
model = configure_gemini()

# --- AI TONES ---
TONE_OPTIONS = {
    "Compassionate Listener": "You are a compassionate listener — soft, empathetic, patient — like a therapist who listens without judgment.",
    "Motivating Coach": "You are a motivating coach — energetic, encouraging, and action-focused — helping the user push through rough days.",
    "Wise Friend": "You are a wise friend — thoughtful, poetic, and reflective — giving soulful responses and timeless advice.",
    "Neutral Therapist": "You are a neutral therapist — balanced, logical, and non-intrusive — asking guiding questions using CBT techniques.",
    "Mindfulness Guide": "You are a mindfulness guide — calm, slow, and grounding — focused on breathing, presence, and awareness."
}
def get_tone_prompt():
    return TONE_OPTIONS.get(st.session_state.get("selected_tone"), TONE_OPTIONS["Compassionate Listener"])

# --- RENDER SIDEBAR ---
render_sidebar()

# --- PAGE ROUTING CONTAINER ---
main_area = st.container()

# --- CONVERSATION INIT ---
if not st.session_state.conversations:
    saved_conversations = load_conversations()
    if saved_conversations:
        st.session_state.conversations = saved_conversations
        if st.session_state.active_conversation == -1:
            st.session_state.active_conversation = 0
    else:
        create_new_conversation()
        st.session_state.active_conversation = 0
    st.rerun()

# --- FEATURE CARDS ---
def render_feature_cards():
    st.markdown(f"""
    <div class="hero-welcome-section">
        <div class="hero-content">
            <h1 class="hero-title">Welcome to TalkHeal, {st.session_state.user_name}! 💬</h1>
            <p class="hero-subtitle">Your Mental Health Companion 💙</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    cols = st.columns(6)
    features = [
        ("🧘‍♀️ Start Yoga", "pages/Yoga.py", "🧘‍♀️", "Yoga & Meditation"),
        ("🌬️ Start Breathing", "pages/Breathing_Exercise.py", "🌬️", "Breathing Exercises"),
        ("📝 Open Journal", "pages/Journaling.py", "📝", "Personal Journaling"),
        ("👨‍⚕️ Find Specialists", "pages/doctor_spec.py", "👨‍⚕️", "Doctor Specialist"),
        ("🛠️ Explore Tools", "pages/selfHelpTools.py", "🛠️", "Self-Help Tools"),
        ("🌿 Open Wellness Hub", "pages/WellnessResourceHub.py", "🌿", "Wellness Hub")
    ]
    for col, (btn_text, page, icon, title) in zip(cols, features):
        with col:
            st.markdown(f"<div class='feature-card'><div class='card-icon'>{icon}</div><h3>{title}</h3></div>", unsafe_allow_html=True)
            if st.button(btn_text, use_container_width=True):
                st.switch_page(page)

# --- GESTURE INPUT MODE ---
# def gesture_mode():
#     st.title("🖐 Gesture Input Mode")
#     if "gesture_active" not in st.session_state:
#         st.session_state.gesture_active = False

#     start_button = st.button("▶️ Start Gesture Mode", disabled=st.session_state.gesture_active)
#     stop_button = st.button("⏹ Stop Gesture Mode", disabled=not st.session_state.gesture_active)
#     FRAME_WINDOW = st.image([])

#     if start_button: st.session_state.gesture_active = True
#     if stop_button: st.session_state.gesture_active = False

#     cap = cv2.VideoCapture(0)
#     while st.session_state.gesture_active:
#         ret, frame = cap.read()
#         if not ret:
#             st.warning("⚠️ Unable to access webcam.")
#             break
#         frame = cv2.flip(frame, 1)
#         prediction = predict_frame(frame)
#         if prediction:
#             st.write(f"✋ Detected Gesture: **{prediction}**")
#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         FRAME_WINDOW.image(frame)
#     cap.release()

# --- ROUTING ---
if st.session_state.show_emergency_page:
    with main_area: render_emergency_page()
elif st.session_state.show_focus_session:
    with main_area: render_focus_session()
elif st.session_state.show_mood_dashboard:
    with main_area: render_mood_dashboard()
else:
    with main_area:
        # 🔹 Add Mode Selector Here
        mode = st.radio("Choose Mode", ["💬 Chat Mode", "🖐 Gesture Mode"], horizontal=True)

        if mode == "💬 Chat Mode":
            render_feature_cards()

            # --- AI Tone ---
            with st.expander("🧠 Customize Your AI Companion"):
                st.markdown("**Choose how your AI companion should respond to you:**")
                selected_tone = st.selectbox(
                    "Select AI personality:",
                    list(TONE_OPTIONS.keys()),
                    index=list(TONE_OPTIONS.keys()).index(st.session_state.selected_tone)
                )
                if selected_tone != st.session_state.selected_tone:
                    st.session_state.selected_tone = selected_tone
                    st.rerun()
                st.info(f"**Current Style**: {TONE_OPTIONS[selected_tone]}")

            # --- Mood Tracking ---
            st.markdown("### 😊 How are you feeling today?")
            mood_options = ['Very Sad 😢', 'Sad 😔', 'Neutral 😐', 'Happy 😊', 'Very Happy 😄']
            mood = st.slider("Select your current mood", 1, 5, 3, 1)
            tips = {
                1: "🤗 It's okay to feel this way. Try some deep breathing.",
                2: "📝 Write down your thoughts in your journal.",
                3: "🚶‍♀️ Take a short walk or stretch.",
                4: "✨ You're happy! Share something positive.",
                5: "🌟 You're shining! Spread positivity."
            }
            col_m, col_t = st.columns([1, 2])
            with col_m:
                st.markdown(f"**Current mood**: {mood_options[mood-1]}")
            with col_t:
                st.info(tips.get(mood))

            st.markdown("---")

            # --- Chatbot ---
            render_chat_interface()
            handle_chat_input(model, system_prompt=get_tone_prompt())
            render_session_controls()

        elif mode == "🖐 Gesture Mode":
            gesture_mode()

# --- AUTO SCROLL CHAT ---
st.markdown("""
<script>
function scrollToBottom() {
    var chatContainer = document.querySelector('.chat-container');
    if (chatContainer) { chatContainer.scrollTop = chatContainer.scrollHeight; }
}
setTimeout(scrollToBottom, 100);
</script>
""", unsafe_allow_html=True)