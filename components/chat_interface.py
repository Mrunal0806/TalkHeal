import streamlit as st
import streamlit.components.v1 as components
from datetime import datetime
from core.utils import get_current_time, get_ai_response, save_conversations
import requests
import textwrap

<<<<<<< HEAD
# Ensures essential session state variables exist with default values to prevent errors
if "pinned_messages" not in st.session_state:
    st.session_state.pinned_messages = []

if "active_conversation" not in st.session_state:
    st.session_state.active_conversation = -1

if "conversations" not in st.session_state:
    st.session_state.conversations = []

=======
>>>>>>> 7aad9a9 (Clean commit: add project files, ignore env & checkpoints)
# Custom CSS for enhanced button styling
def inject_custom_css():
    st.markdown("""
    <style>
    /* Enhanced button styling for session controls */
    .stButton > button {
        background: linear-gradient(135deg, #FF6B9D 0%, #C44569 100%) !important;
        color: white !important;
        border: 2px solid rgba(255,255,255,0.2) !important;
        border-radius: 15px !important;
        padding: 10px 16px !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        box-shadow: 0 6px 20px rgba(255,107,157,0.3), 0 3px 10px rgba(0,0,0,0.1) !important;
        transition: all 0.3s ease !important;
        position: relative !important;
        overflow: hidden !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(255,182,193,0.45), 0 4px 15px rgba(0,0,0,0.15) !important;
        background: linear-gradient(135deg, #ffd1dc 0%, #ff9aa2 100%) !important; /* light pink hover */
    }
    /* Enhanced session control buttons */
    .stButton > button {
        background: linear-gradient(135deg, #C2185B 0%, #FCBBF4 100%) !important;
        color: white !important;
        font-weight: 700 !important;
        font-size: 16px !important;
        border-radius: 20px !important;
        padding: 15px 25px !important;
        box-shadow: 0 8px 25px rgba(194,24,91,0.4), 0 4px 15px rgba(0,0,0,0.2) !important;
        border: 3px solid rgba(255,255,255,0.3) !important;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3) !important;
        transition: all 0.3s ease !important;
        min-height: 60px !important;
    }
    .stButton > button:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 12px 35px rgba(194,24,91,0.5), 0 6px 20px rgba(0,0,0,0.25) !important;
        background: linear-gradient(135deg, #E91E63 0%, #9C27B0 100%) !important;
        border-color: rgba(255,255,255,0.5) !important;
    }

    /* Specific color for Session Summary and End Chat buttons (override any glass styles) */
    #session-controls .stButton > button[data-testid="baseButton-session_summary"],
    #session-controls .stButton > button[data-testid="baseButton-end_chat"] {
        background: #C2185B !important;
        background-color: #C2185B !important;
        background-image: none !important;
        color: #ffffff !important;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3) !important;
        border: 3px solid rgba(255,255,255,0.3) !important;
        box-shadow: 0 6px 20px rgba(194,24,91,0.4), 0 3px 10px rgba(0,0,0,0.15) !important;
        backdrop-filter: none !important;
        -webkit-backdrop-filter: none !important;
        filter: none !important;
        opacity: 1 !important;
    }
    #session-controls .stButton > button[data-testid="baseButton-session_summary"]:hover,
    #session-controls .stButton > button[data-testid="baseButton-end_chat"]:hover {
        background: #E91E63 !important;
        background-color: #E91E63 !important;
        background-image: none !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(194,24,91,0.5), 0 4px 15px rgba(0,0,0,0.2) !important;
        border-color: rgba(255,255,255,0.5) !important;
    }
    /* Force solid red on specific buttons by key (prefix match) */
    .stButton > button[data-testid^="baseButton-session_summary"],
    .stButton > button[data-testid^="baseButton-end_chat"] {
        background: #C2185B !important;
        background-color: #C2185B !important;
        background-image: none !important;
        color: #ffffff !important;
        border: 3px solid rgba(255,255,255,0.3) !important;
        box-shadow: 0 6px 20px rgba(194,24,91,0.4), 0 3px 10px rgba(0,0,0,0.15) !important;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3) !important;
        opacity: 1 !important;
    }
    .stButton > button[data-testid^="baseButton-session_summary"]:hover,
    .stButton > button[data-testid^="baseButton-end_chat"]:hover {
        background: #E91E63 !important;
        background-color: #E91E63 !important;
        background-image: none !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(194,24,91,0.5), 0 4px 15px rgba(0,0,0,0.2) !important;
        border-color: rgba(255,255,255,0.5) !important;
    }
    /* Feedback buttons: solid dark pink, uniform small size */
    .stButton > button[data-testid^="baseButton-feedback_"],
    .stButton > button[data-testid^="baseButton-during_feedback_"] {
        background: #C2185B !important; /* solid dark pink */
        color: white !important;
        font-weight: 700 !important;
        font-size: 18px !important;
        border-radius: 14px !important;
        width: 190px !important;   /* consistent small width */
        height: 56px !important;   /* consistent small height */
        box-shadow: 0 6px 20px rgba(194,24,91,0.4), 0 3px 10px rgba(0,0,0,0.15) !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        border: 3px solid rgba(255,255,255,0.3) !important;
        line-height: 1.1 !important;
        white-space: nowrap !important; /* single line */
        margin: 0 auto 8px auto !important;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3) !important;
        transition: all 0.3s ease !important;
    }
    /* Hover effect for feedback buttons */
    .stButton > button[data-testid^="baseButton-feedback_"]:hover,
    .stButton > button[data-testid^="baseButton-during_feedback_"]:hover {
        background: #E91E63 !important; /* lighter pink on hover */
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(194,24,91,0.5), 0 4px 15px rgba(0,0,0,0.2) !important;
        border-color: rgba(255,255,255,0.5) !important;
    }
    /* Make the 'Okay' button a bit wider for visual balance */
    .stButton > button[data-testid="baseButton-feedback_1"],
    .stButton > button[data-testid="baseButton-during_feedback_1"] {
        width: 250px !important;
    }
    .stButton > button[data-testid^="baseButton-feedback_"] span,
    .stButton > button[data-testid^="baseButton-during_feedback_"] span {
        font-size: 20px !important; /* keep emoji/text reasonable */
        display: inline-block !important;
    }
    /* Force red scheme for primary buttons (used by session controls) */
    .stButton > button[kind="primary"] {
        background: #C2185B !important;
        background-color: #C2185B !important;
        background-image: none !important;
        color: #ffffff !important;
        border: 3px solid rgba(255,255,255,0.3) !important;
        box-shadow: 0 6px 20px rgba(194,24,91,0.4), 0 3px 10px rgba(0,0,0,0.15) !important;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3) !important;
    }
    .stButton > button[kind="primary"]:hover {
        background: #E91E63 !important;
        background-color: #E91E63 !important;
        background-image: none !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(194,24,91,0.5), 0 4px 15px rgba(0,0,0,0.2) !important;
        border-color: rgba(255,255,255,0.5) !important;
    }

    .chat-container {
        max-width: 1000px;
        margin: 0 auto;
        padding: 15px;
        display: flex;
        flex-direction: column;
    }

    .user-message, .bot-message {
        max-width: 65%;
        padding: 12px 16px;
        border-radius: 16px;
        margin: 8px 0;
        word-wrap: break-word;
        font-size: 15px;
        line-height: 1.5;
    }

    /* User bubble */
    .user-message {
        color: #fff;
        background: linear-gradient(130deg, #6366f1 70%, #818cf8 100%);
        border: 1.5px solid rgba(129,140,248,0.21);
        align-self: flex-end;
        border-bottom-right-radius: 4px;
    }

    /* Bot bubble */
    .bot-message {
        background: var(--glass-effect);
        background-color: var(--surface-alt);
        color: #efeef9;
        align-self: flex-start;
        border-bottom-left-radius: 4px;
    }

    /* Timestamp */
    .message-time {
        font-size: 12px;
        color: #c4d0e0;
        opacity: .76;
        margin-top: 4px;
        text-align: right;
    }

    </style>
    """, unsafe_allow_html=True)


# Inject JS to get user's local time zone
def set_user_time_in_session():
    if "user_time_offset" not in st.session_state:
        components.html("""
            <script>
            const offset = new Date().getTimezoneOffset(); 
            const time = new Date().toLocaleString();      
            const data = {offset: offset, time: time};
            window.parent.postMessage({type: 'USER_TIME', data: data}, '*');
            </script>
        """, height=0)

        st.markdown("""
        <script>
        window.addEventListener("message", (event) => {
            if (event.data.type === "USER_TIME") {
                const payload = JSON.stringify(event.data.data);
                fetch("/", {
                    method: "POST",
                    headers: {"Content-Type": "application/json"},
                    body: payload
                }).then(() => location.reload());
            }
        });
        </script>
        """, unsafe_allow_html=True)

#set_user_time_in_session()

def show_session_feedback():
    st.markdown("---")
    st.subheader("💬 Session Feedback & Emotional Closure Message")

    # Emotional closure with enhanced messaging
    st.success("🌟 **Thank you for sharing your thoughts today!** 🌟")
    st.markdown("""
    <div style="
        background:linear-gradient(135deg, #FFE4EF 0%, #FFD1E8 100%);
        padding:28px; border-radius:24px; margin:24px 0;
        box-shadow:0 12px 36px rgba(255,107,157,0.25), 0 6px 18px rgba(0,0,0,0.08);
        border:2px solid rgba(255,107,157,0.35);
        position:relative; overflow:hidden;">
        <div style="position:absolute; top:-22px; right:-22px; width:px; height:72px; background:linear-gradient(45deg, #FF6B9D, #FE8BBE); border-radius:50%; opacity:0.15;"></div>
        <div style="position:absolute; bottom:-26px; left:-26px; width:96px; height:96px; background:linear-gradient(45deg, #FE8BBE, #FF6B9D); border-radius:50%; opacity:0.12;"></div>
        <h4 style="color:#C2185B; margin-bottom:14px; font-size:22px; font-weight:700; letter-spacing:0.2px; text-shadow:0 2px 6px rgba(0,0,0,0.08);">💗 Your mental health journey matters to us</h4>
        <p style="color:#5C5C5C; line-height:1.7; font-size:16px; margin:0; text-shadow:0 1px 2px rgba(0,0,0,0.05);">
            We hope our conversation today provided you with comfort, support, and valuable insights.
            Remember, it's okay to not be okay, and seeking help is a sign of strength.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Enhanced Problem / Solution / Technical Suggestions / Expected Impact
    st.markdown("### 📌 Session Summary & Platform Impact")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🔍 Problem Identified:**")
        st.markdown("""
        <div style="background:rgba(255,255,255,0.72); backdrop-filter:blur(10px); -webkit-backdrop-filter:blur(10px); padding:18px; border-radius:16px; margin:10px 0; box-shadow:0 8px 24px rgba(0,0,0,0.10); border:1px solid rgba(255,255,255,0.55); border-left:5px solid #64748B;">
            <p style="color:#1F2937; margin:0; font-size:15px; line-height:1.6; font-weight:500;">Users need emotional closure and a way to provide feedback about their session experience to improve the platform.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("**💡 Solution Provided:**")
        st.markdown("""
        <div style="background:rgba(255,255,255,0.72); backdrop-filter:blur(10px); -webkit-backdrop-filter:blur(10px); padding:18px; border-radius:16px; margin:10px 0; box-shadow:0 8px 24px rgba(0,0,0,0.10); border:1px solid rgba(255,255,255,0.55); border-left:5px solid #64748B;">
            <p style="color:#1F2937; margin:0; font-size:15px; line-height:1.6; font-weight:500;">Implemented comprehensive session feedback system with emoji reactions and emotional closure messaging.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("**⚙️ Technical Implementation:**")
        st.markdown("""
        <div style="background:rgba(255,255,255,0.72); backdrop-filter:blur(10px); -webkit-backdrop-filter:blur(10px); padding:18px; border-radius:16px; margin:10px 0; box-shadow:0 8px 24px rgba(0,0,0,0.10); border:1px solid rgba(255,255,255,0.55); border-left:5px solid #64748B;">
            <ul style="color:#1F2937; margin:0; padding-left:20px; font-size:15px; line-height:1.6; font-weight:500;">
                <li>Interactive emoji feedback system</li>
                <li>Session state management</li>
                <li>User experience tracking</li>
                <li>Emotional closure algorithms</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("**🎯 Expected Impact:**")
        st.markdown("""
        <div style="background:rgba(255,255,255,0.72); backdrop-filter:blur(10px); -webkit-backdrop-filter:blur(10px); padding:18px; border-radius:16px; margin:10px 0; box-shadow:0 8px 24px rgba(0,0,0,0.10); border:1px solid rgba(255,255,255,0.55); border-left:5px solid #64748B;">
            <ul style="color:#1F2937; margin:0; padding-left:20px; font-size:15px; line-height:1.6; font-weight:500;">
                <li>Smoother Higher user satisfaction</li>
                <li>Better emotional well-being</li>
                <li>Improved platform engagement</li>
                <li>Data-driven improvements</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # Enhanced Emoji feedback with better UX
    st.markdown("### 🌟 How was your session experience?")
    
    # Create a more engaging feedback interface
    feedback_options = [
        ("😊", "Excellent"),
        ("😐", "Okay"),
        ("🙁", "Needs Help"),
        ("💪", "Empowering")
    ]
    
    selected_feedback = None
    feedback_cols = st.columns(len(feedback_options))
    for i, (emoji, description) in enumerate(feedback_options):
        with feedback_cols[i]:
            btn_label = f"{emoji} {description}"
            if st.button(btn_label, key=f"feedback_{i}"):
                selected_feedback = (emoji, description)
                st.session_state.last_feedback = selected_feedback

    # Show feedback confirmation and emotional closure
    if selected_feedback or st.session_state.get("last_feedback"):
        feedback_to_show = selected_feedback or st.session_state.get("last_feedback")
        emoji, description = feedback_to_show
        
        st.markdown("---")
        st.markdown(f"### {emoji} Thank you for your feedback!")
        st.success(f"**Feedback received:** {description}")
        
        # Enhanced Emotional Closure Message
        st.markdown("### 💬 Emotional Closure & Platform Impact")
        
        closure_messages = {
            "😊": "Your positive experience motivates us to continue providing compassionate mental health support.",
            "😐": "We're committed to improving your experience. Your feedback helps us grow.",
            "🙁": "We're sorry your experience wasn't ideal. We're working to make it better.",
            "🤗": "Your amazing feedback inspires our team to keep innovating mental health support.",
            "💪": "Empowering you is our mission. Thank you for trusting us with your journey."
        }
        
        st.info(closure_messages.get(emoji, "Thank you for being part of our mental health community."))
        
        # Platform impact summary with TalkHeal-themed styling
        st.markdown("""
        <div style="background:linear-gradient(135deg, #E91E63 0%, #BA68C8 50%, #FF5722 100%); color:white; padding:20px; border-radius:15px; margin:20px 0; box-shadow:0 8px 32px rgba(233,30,99,0.3); border:2px solid rgba(255,255,255,0.2);">
            <h5 style="color:#ffffff; margin-bottom:15px; font-size:18px; text-align:center;">📊 Platform Impact Summary</h5>
            <ul style="color:#ffffff; list-style-type:none; padding:0;">
                <li style="margin:12px 0; padding:8px 12px; background:rgba(255,255,255,0.15); border-radius:8px; border-left:4px solid #FF69B4;"><strong style="color:#FFD700;">User Experience:</strong> <span style="color:#ffffff;">Enhanced emotional closure and feedback collection</span></li>
                <li style="margin:12px 0; padding:8px 12px; background:rgba(255,255,255,0.15); border-radius:8px; border-left:4px solid #FF1493;"><strong style="color:#FFD700;">Data Insights:</strong> <span style="color:#ffffff;">Better understanding of user satisfaction and needs</span></li>
                <li style="margin:12px 0; padding:8px 12px; background:rgba(255,255,255,0.15); border-radius:8px; border-left:4px solid #C71585;"><strong style="color:#FFD700;">Platform Growth:</strong> <span style="color:#ffffff;">Improved engagement and user retention</span></li>
                <li style="margin:12px 0; padding:8px 12px; background:rgba(255,255,255,0.15); border-radius:8px; border-left:4px solid #FF69B4;"><strong style="color:#FFD700;">Mental Health Support:</strong> <span style="color:#ffffff;">More personalized and effective assistance</span></li>
            </ul>
        </div>
        """, unsafe_allow_html=True)


def show_session_feedback_during_session():
    """Show session feedback during an active session"""
    st.markdown("### 🌟 How is your session going so far?")
    
    # Quick feedback options for during session
    feedback_options = [
        ("😊", "Excellent"),
        ("😐", "Okay"),
        ("🙁", "Needs Help"),
        ("💪", "Empowering")
    ]
    
    selected_feedback = None
    feedback_cols = st.columns(len(feedback_options))
    for i, (emoji, description) in enumerate(feedback_options):
        with feedback_cols[i]:
            btn_label = f"{emoji} {description}"
            if st.button(btn_label, key=f"during_feedback_{i}"):
                selected_feedback = (emoji, description)
                st.session_state.during_session_feedback = selected_feedback

    if selected_feedback or st.session_state.get("during_session_feedback"):
        feedback_to_show = selected_feedback or st.session_state.get("during_session_feedback")
        emoji, description = feedback_to_show
        
        st.success(f"**Feedback received:** {description}")
        st.info("Thank you for your feedback! We're here to support you throughout your session.")


def show_session_summary(active_convo):
    """Show a summary of the current session"""
    if not active_convo["messages"]:
        st.info("No messages in this session yet.")
        return
    
    # Count message types
    user_messages = len([msg for msg in active_convo["messages"] if msg["sender"] == "user"])
    bot_messages = len([msg for msg in active_convo["messages"] if msg["sender"] == "bot"])
    total_messages = len(active_convo["messages"])

    # Visually appealing stat boxes with light pink background, border, and larger font
    st.markdown("""
    <div style="display:flex; gap:32px; justify-content:center; margin-bottom:32px; flex-wrap:wrap;">
        <div style="background:rgba(255,182,193,0.35); box-shadow:0 8px 32px rgba(255,182,193,0.25), 0 2px 8px rgba(0,0,0,0.10); border-radius:22px; border:2px solid #ffb6c1; padding:32px 44px; min-width:200px; text-align:center;">
            <div style="font-size:22px; font-weight:700; color:#b8005c; letter-spacing:1px; margin-bottom:12px;">Total Messages</div>
            <div style="font-size:54px; font-weight:900; color:#b8005c; text-shadow:0 2px 8px rgba(0,0,0,0.10);">{total_messages}</div>
        </div>
        <div style="background:rgba(255,182,193,0.35); box-shadow:0 8px 32px rgba(255,182,193,0.25), 0 2px 8px rgba(0,0,0,0.10); border-radius:22px; border:2px solid #ffb6c1; padding:32px 44px; min-width:200px; text-align:center;">
            <div style="font-size:22px; font-weight:700; color:#b8005c; letter-spacing:1px; margin-bottom:12px;">Your Messages</div>
            <div style="font-size:54px; font-weight:900; color:#b8005c; text-shadow:0 2px 8px rgba(0,0,0,0.10);">{user_messages}</div>
        </div>
        <div style="background:rgba(255,182,193,0.35); box-shadow:0 8px 32px rgba(255,182,193,0.25), 0 2px 8px rgba(0,0,0,0.10); border-radius:22px; border:2px solid #ffb6c1; padding:32px 44px; min-width:200px; text-align:center;">
            <div style="font-size:22px; font-weight:700; color:#b8005c; letter-spacing:1px; margin-bottom:12px;">AI Responses</div>
            <div style="font-size:54px; font-weight:900; color:#b8005c; text-shadow:0 2px 8px rgba(0,0,0,0.10);">{bot_messages}</div>
        </div>
    </div>
    """.format(total_messages=total_messages, user_messages=user_messages, bot_messages=bot_messages), unsafe_allow_html=True)

    # Session insights
    st.markdown("### 📈 Session Insights")
    if user_messages > 0:
        st.info(f"**Session Duration:** Active conversation with {user_messages} exchanges")
        st.info(f"**Engagement Level:** {'High' if user_messages > 5 else 'Medium' if user_messages > 2 else 'Starting'} engagement")
    
    # Recent conversation preview
    if len(active_convo["messages"]) > 0:
        st.markdown("### 💭 Recent Conversation")
        recent_messages = active_convo["messages"][-3:]  # Show last 3 messages
        for msg in recent_messages:
            sender_icon = "👤" if msg["sender"] == "user" else "🤖"
            st.markdown(f"{sender_icon} **{msg['sender'].title()}:** {msg['message'][:100]}{'...' if len(msg['message']) > 100 else ''}")

<<<<<<< HEAD
# Functions to handle pinning/unpinning messages and rendering the chat interface with pin buttons
#Adds/removes a message from pinned messages in session state
def toggle_pin_message(msg, convo_id):
    """Add or remove a message from pinned messages"""
    if "pinned_messages" not in st.session_state:
        st.session_state.pinned_messages = []

    # Unique check based on message text + convo_id
    exists = next(
        (m for m in st.session_state.pinned_messages
         if m.get("message") == msg.get("message")
         and m.get("convo_id") == convo_id),
        None
    )

    if exists:
        # Unpin by removing
        st.session_state.pinned_messages = [
            m for m in st.session_state.pinned_messages if m != exists
        ]
    else:
        # Save only what you need
        st.session_state.pinned_messages.append({
            "message": msg.get("message", ""),
            "sender": msg.get("sender", ""),   # keep if you want filtering
            "convo_id": convo_id
        })
        
# Displays chat messages with styled bubbles and pin/unpin functionality
def render_chat_interface():
    inject_custom_css()

    if st.session_state.active_conversation >= 0:
        active_convo = st.session_state.conversations[st.session_state.active_conversation]

        if not active_convo["messages"]:
            st.markdown(f"""
<div class="welcome-message" style="padding:12px; border-radius:10px; background-color: #f0f0f0; margin-bottom: 20px;">
    <strong>Hello! I'm TalkHeal, your mental health companion 🤗</strong><br>
    How are you feeling today? You can write below or start a new topic.
    <div class="message-time" style="font-size:10px; text-align:right; margin-top: 8px;">{get_current_time()}</div>
</div>
            """, unsafe_allow_html=True)

        # Start the chat container (no fixed max-width wrapper)
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)

        for i, msg in enumerate(active_convo["messages"]):
            # Check if this message is pinned
            pinned = any(
                m["message"] == msg["message"]
                and m.get("convo_id") == st.session_state.active_conversation
                for m in st.session_state.pinned_messages
            )
            pin_label = "📍" if pinned else "📌"

            if msg["sender"] == "user":
                # User message aligned right
                col1, col2, col3 = st.columns([2, 7, 1])
                with col2:
                    st.markdown(f"""
<div class="user-message" style="
    background: linear-gradient(130deg, #6366f1 70%, #818cf8 100%);
    color: white;
    padding: 12px 16px;
    border-radius: 16px;
    margin: 8px 0;
    border: 1.5px solid rgba(129,140,248,0.21);
    border-bottom-right-radius: 4px;
    word-wrap: break-word;
    font-size: 15px;
    line-height: 1.5;
    position: relative;
    margin-left: auto;
    max-width: 85%;
">
    {msg['message']}
    <div class="message-time" style="font-size:12px; color: #c4d0e0; opacity: 0.76; margin-top: 4px; text-align: right;">
        {msg['time']}
    </div>
</div>
                    """, unsafe_allow_html=True)
                with col3:
                    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
                    if st.button(pin_label, key=f"pin_{i}", help="Pin/Unpin this message"):
                        toggle_pin_message(msg, st.session_state.active_conversation)
                        st.rerun()
            else:
                # Bot message aligned left
                col1, col2, col3 = st.columns([1, 7, 2])
                with col1:
                    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
                    if st.button(pin_label, key=f"pin_{i}", help="Pin/Unpin this message"):
                        toggle_pin_message(msg, st.session_state.active_conversation)
                        st.rerun()
                with col2:
                    st.markdown(f"""
<div class="bot-message" style="
    background: rgba(255,255,255,0.9);
    color: #333;
    padding: 12px 16px;
    border-radius: 16px;
    margin: 8px 0;
    border: 1px solid rgba(0,0,0,0.1);
    border-bottom-left-radius: 4px;
    word-wrap: break-word;
    font-size: 15px;
    line-height: 1.5;
    position: relative;
    margin-right: auto;
    max-width: 85%;
">
    {msg['message']}
    <div class="message-time" style="font-size:12px; color: #666; opacity: 0.76; margin-top: 4px; text-align: right;">
        {msg['time']}
    </div>
</div>
                    """, unsafe_allow_html=True)

        # Close chat container
        st.markdown('</div>', unsafe_allow_html=True)
        
# Quickly lists all pinned messages for reference
def render_pinned_messages():
    if st.session_state.pinned_messages:
        st.markdown("### 📌 Pinned Messages")
        for msg in st.session_state.pinned_messages:
            st.markdown(f"{msg['message']}")
=======

# Display chat messages
def render_chat_interface():    
    # Inject custom CSS for enhanced button styling
    inject_custom_css()
    
    if st.session_state.active_conversation >= 0:
        active_convo = st.session_state.conversations[st.session_state.active_conversation]
        
        if not active_convo["messages"]:
            st.markdown(f"""
<div class="welcome-message">
    <strong>Hello! I'm TalkHeal, your mental health companion 🤗</strong><br>
    How are you feeling today? You can write below or start a new topic.
    <div class="message-time">{get_current_time()}</div>
</div>
            """, unsafe_allow_html=True)
        
        chat_html = '<div class="chat-container">'

        for msg in active_convo["messages"]:
            css_class = "user-message" if msg["sender"] == "user" else "bot-message"
            chat_html += f"""
<div class="{css_class}">
    {msg["message"]}
    <div class="message-time">{msg["time"]}</div>
</div>
"""
        chat_html += '</div>'
        st.markdown(chat_html, unsafe_allow_html=True)

>>>>>>> 7aad9a9 (Clean commit: add project files, ignore env & checkpoints)


def render_session_controls():
    """Render session controls and feedback - to be called after chat input"""
    if st.session_state.active_conversation >= 0:
        active_convo = st.session_state.conversations[st.session_state.active_conversation]
        
        # Show session controls below chat input
        if not active_convo.get("session_ended", False):
            st.markdown("---")
            st.markdown('<div id="session-controls">', unsafe_allow_html=True)
            col1, col2, col3 = st.columns([1, 1, 1])
            
            with col1:
                if st.button("📊 Session Summary", key="session_summary", help="View a summary of your current session", type="primary"):
                    st.session_state.show_session_summary = True
                    st.rerun()
            
            with col2:
                if st.button("End Chat Session", key="end_chat", type="primary"):
                    active_convo["session_ended"] = True
                    st.session_state.conversations[st.session_state.active_conversation] = active_convo
                    st.rerun()  # Refresh UI immediately
            
            with col3:
                # Placeholder for balance
                pass
            
            # Show session summary if requested
            if st.session_state.get("show_session_summary", False):
                st.markdown("---")
                st.subheader("📊 Current Session Summary")
                show_session_summary(active_convo)
                
                if st.button("✖️ Close Summary", key="close_summary"):
                    st.session_state.show_session_summary = False
                    st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Display enhanced feedback section if session ended
        if active_convo.get("session_ended", False):
            st.markdown("---")
            components.html(textwrap.dedent("""
<div style="background:linear-gradient(135deg, #FF6B9D 0%, #C44569 50%, #FF8E53 100%); color:white; padding:30px; border-radius:25px; margin:25px 0; text-align:center; box-shadow:0 15px 50px rgba(255,107,157,0.4), 0 8px 30px rgba(0,0,0,0.2); border:3px solid rgba(255,255,255,0.3); position:relative; overflow:hidden;">
    <div style="position:absolute; top:-30px; left:-30px; width:80px; height:80px; background:linear-gradient(45deg, rgba(255,255,255,0.1), rgba(255,255,255,0.05)); border-radius:50%;"></div>
    <div style="position:absolute; bottom:-30px; right:-30px; width:60px; height:60px; background:linear-gradient(45deg, rgba(255,255,255,0.08), rgba(255,255,255,0.03)); border-radius:50%;"></div>

    <h3 style="font-size:28px; margin-bottom:20px; text-shadow:0 4px 8px rgba(0,0,0,0.3); font-weight:700;">🌟 Session Complete! 🌟</h3>
    <p style="font-size:20px; margin:15px 0; font-weight:600; text-shadow:0 2px 4px rgba(0,0,0,0.2);">💚 <strong>Loved your input in this session!.</strong></p>
    <p style="font-size:18px; margin:10px 0; opacity:0.95; text-shadow:0 1px 2px rgba(0,0,0,0.2);"><em>Your session has ended. You can start a new one anytime.</em></p>

    <div style="margin-top:25px; display:flex; justify-content:center; gap:15px; flex-wrap:wrap;">
        <span style="background:rgba(255,255,255,0.25); padding:12px 20px; border-radius:25px; margin:5px; font-size:20px; box-shadow:0 4px 15px rgba(0,0,0,0.2); border:2px solid rgba(255,255,255,0.3); transition:all 0.3s ease;" onmouseover="this.style.transform='scale(1.1)'; this.style.background='rgba(255,255,255,0.35)'" onmouseout="this.style.transform='scale(1)'; this.style.background='rgba(255,255,255,0.25)'">😊</span>
        <span style="background:rgba(255,255,255,0.25); padding:12px 20px; border-radius:25px; margin:5px; font-size:20px; box-shadow:0 4px 15px rgba(0,0,0,0.2); border:2px solid rgba(255,255,255,0.3); transition:all 0.3s ease;" onmouseover="this.style.transform='scale(1.1)'; this.style.background='rgba(255,255,255,0.35)'" onmouseout="this.style.transform='scale(1)'; this.style.background='rgba(255,255,255,0.25)'">😐</span>
        <span style="background:rgba(255,255,255,0.25); padding:12px 20px; border-radius:25px; margin:5px; font-size:20px; box-shadow:0 4px 15px rgba(0,0,0,0.2); border:2px solid rgba(255,255,255,0.3); transition:all 0.3s ease;" onmouseover="this.style.transform='scale(1.1)'; this.style.background='rgba(255,255,255,0.35)'" onmouseout="this.style.transform='scale(1)'; this.style.background='rgba(255,255,255,0.25)'">🙁</span>
        <span style="background:rgba(255,255,255,0.25); padding:12px 20px; border-radius:25px; margin:5px; font-size:20px; box-shadow:0 4px 15px rgba(0,0,0,0.2); border:2px solid rgba(255,255,255,0.3); transition:all 0.3s ease;" onmouseover="this.style.transform='scale(1.1)'; this.style.background='rgba(255,255,255,0.35)'" onmouseout="this.style.transform='scale(1)'; this.style.background='rgba(255,255,255,0.25)'">🤗</span>
        <span style="background:rgba(255,255,255,0.25); padding:12px 20px; border-radius:25px; margin:5px; font-size:20px; box-shadow:0 4px 15px rgba(0,0,0,0.2); border:2px solid rgba(255,255,255,0.3); transition:all 0.3s ease;" onmouseover="this.style.transform='scale(1.1)'; this.style.background='rgba(255,255,255,0.35)'" onmouseout="this.style.transform='scale(1)'; this.style.background='rgba(255,255,255,0.25)'">💪</span>
    </div>
</div>
"""), height=420)
            
            # Show comprehensive session feedback
            show_session_feedback()


# Handle chat input and generate AI response
def handle_chat_input(model, system_prompt):
    if "pre_filled_chat_input" not in st.session_state:
        st.session_state.pre_filled_chat_input = ""
    initial_value = st.session_state.pre_filled_chat_input
    st.session_state.pre_filled_chat_input = ""

    with st.form(key="chat_form", clear_on_submit=True):
        col1, col2 = st.columns([5, 1])
        with col1:
            user_input = st.text_input(
                "Share your thoughts...",
                key="message_input",
                label_visibility="collapsed",
                placeholder="Type your message here...",
                value=initial_value
            )
        with col2:
            send_pressed = st.form_submit_button("Send", use_container_width=True)

    if (send_pressed or st.session_state.get("send_chat_message", False)) and user_input.strip():
        if 'send_chat_message' in st.session_state:
            st.session_state.send_chat_message = False

        if st.session_state.active_conversation >= 0:
            current_time = get_current_time()
            active_convo = st.session_state.conversations[st.session_state.active_conversation]

            # Save user message
            active_convo["messages"].append({
                "sender": "user",
                "message": user_input.strip(),
                "time": current_time
            })

            # Set title if it's the first message
            if len(active_convo["messages"]) == 1:
                title = user_input[:30] + "..." if len(user_input) > 30 else user_input
                active_convo["title"] = title

            save_conversations(st.session_state.conversations)

            # Format memory
            def format_memory(convo_history, max_turns=10):
                context = ""
                for msg in convo_history[-max_turns*2:]:  # user + bot per turn
                    sender = "User" if msg["sender"] == "user" else "Bot"
                    context += f"{sender}: {msg['message']}\n"
                return context

            try:
                with st.spinner("TalkHeal is thinking..."):
                    memory = format_memory(active_convo["messages"])
                    # Create a comprehensive prompt combining system prompt and conversation context
                    full_prompt = f"{system_prompt}\n\nConversation Context:\n{memory}\n\nUser: {user_input.strip()}"
                    ai_response = get_ai_response(full_prompt, model)

                    active_convo["messages"].append({
                        "sender": "bot",
                        "message": ai_response,
                        "time": get_current_time()
                    })

            except ValueError as e:
                st.error("I'm having trouble understanding your message. Could you please rephrase it?")
                active_convo["messages"].append({
                    "sender": "bot",
                    "message": "I'm having trouble understanding your message. Could you please rephrase it?",
                    "time": get_current_time()
                })
            except requests.RequestException as e:
                st.error("Network connection issue. Please check your internet connection.")
                active_convo["messages"].append({
                    "sender": "bot",
                    "message": "I'm having trouble connecting to my services. Please check your internet connection and try again.",
                    "time": get_current_time()
                })
            except Exception as e:
                st.error(f"An unexpected error occurred. Please try again.")
                active_convo["messages"].append({
                    "sender": "bot",
                    "message": "I'm having trouble responding right now. Please try again in a moment.",
                    "time": get_current_time()
                })

            save_conversations(st.session_state.conversations)
            st.rerun()
            
# This file contains component functions that are imported and used by the main TalkHeal.py
# The main() function and standalone execution code has been removed since this is a component module
