import streamlit as st
import webbrowser
from datetime import datetime
from core.utils import create_new_conversation, get_current_time
from core.theme import get_current_theme, toggle_theme, set_palette, PALETTES
from components.mood_dashboard import render_mood_dashboard_button, MoodTracker
from components.profile import initialize_profile_state, render_profile_section
from streamlit_js_eval import streamlit_js_eval
import requests
import random
from datetime import datetime

# --- Structured Emergency Resources ---
GLOBAL_RESOURCES = [
    {"name": "Befrienders Worldwide", "desc": "Emotional support to prevent suicide worldwide.",
        "url": "https://www.befrienders.org/"},
    {"name": "International Association for Suicide Prevention (IASP)", "desc": "Find a crisis center anywhere in the world.",
     "url": "https://www.iasp.info/resources/Crisis_Centres/"},
    {"name": "Crisis Text Line", "desc": "Text-based support available in the US, UK, Canada, and Ireland.",
     "url": "https://www.crisistextline.org/"},
    {"name": "The Trevor Project", "desc": "Crisis intervention and suicide prevention for LGBTQ young people.",
     "url": "https://www.thetrevorproject.org/"},
    {"name": "Child Helpline International", "desc": "A global network of child helplines for young people in need of help.",
     "url": "https://www.childhelplineinternational.org/"}
]

def get_country_from_coords(lat, lon):
    try:
        url = f"https://geocode.maps.co/reverse?lat={lat}&lon={lon}"
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            return data.get("address", {}).get("country_code", "").upper()
    except:
        pass
    return None

def get_user_country():
    # 1. Try to get user's actual browser location (via JS)
    coords = streamlit_js_eval(
        js_expressions="""
            new Promise((resolve, reject) => {
                navigator.geolocation.getCurrentPosition(
                    position => resolve({
                        latitude: position.coords.latitude,
                        longitude: position.coords.longitude
                    }),
                    error => resolve(null)
                );
            });
        """,
        key="get_coords"
    )

    if coords and "latitude" in coords and "longitude" in coords:
        country = get_country_from_coords(coords["latitude"], coords["longitude"])
        if country:
            return country

    # 2. Fallback to IP-based location using ipapi.co (no key required)
    try:
        resp = requests.get("https://ipapi.co/json/", timeout=3)
        if resp.status_code == 200:
            return resp.json().get("country_code", "").upper()
    except:
        pass

    return None  # final fallback if everything fails

country_helplines = {
    "US": [
        "National Suicide Prevention Lifeline: 988",
        "Crisis Text Line: Text HOME to 741741",
        "SAMHSA National Helpline: 1-800-662-4357"
    ],
    "IN": [
        "AASRA: 9152987821",
        "Sneha Foundation: 044-24640050"
    ],
    "GB": [
        "Samaritans: 116 123"
    ],
    "AU": [
        "Lifeline: 13 11 14"
    ]
}
IASP_LINK = "https://findahelpline.com/"

mental_health_resources_full = {
    "Depression & Mood Disorders": {
        "description": "Information on understanding and coping with depression, persistent depressive disorder, and other mood-related challenges.",
        "links": [
            {"label": "NIMH - Depression",
                "url": "https://www.nimh.nih.gov/health/topics/depression"},
            {"label": "Mayo Clinic - Depression",
                "url": "https://www.mayoclinic.org/diseases-conditions/depression/symptoms-causes/syc-20356007"}
        ]
    },
    "Anxiety & Panic Disorders": {
        "description": "Guidance on managing generalized anxiety, social anxiety, panic attacks, and phobias.",
        "links": [
            {"label": "ADAA - Anxiety & Depression", "url": "https://adaa.org/"},
            {"label": "NIMH - Anxiety Disorders",
                "url": "https://www.nimh.nih.gov/health/topics/anxiety-disorders"}
        ]
    },
    "Bipolar Disorder": {
        "description": "Understanding the complexities of bipolar disorder, including mood swings and treatment options.",
        "links": [
            {"label": "NIMH - Bipolar Disorder",
                "url": "https://www.nimh.nih.gov/health/topics/bipolar-disorder"}
        ]
    },
    "PTSD & Trauma": {
        "description": "Resources for individuals experiencing post-traumatic stress disorder and other trauma-related conditions.",
        "links": [
            {"label": "PTSD: National Center", "url": "https://www.ptsd.va.gov/"}
        ]
    },
    "OCD & Related Disorders": {
        "description": "Support and information for obsessive-compulsive disorder, body dysmorphic disorder, and hoarding disorder.",
        "links": [
            {"label": "IOCDF - OCD", "url": "https://iocdf.org/"}
        ]
    },
    "Coping Skills & Self-Care": {
        "description": "Practical strategies and techniques for stress management, emotional regulation, and daily well-being.",
        "links": [
            {"label": "HelpGuide - Stress Management",
                "url": "https://www.helpguide.org/articles/stress/stress-management.htm"}
        ]
    },
    "Therapy & Treatment Options": {
        "description": "Overview of various therapeutic approaches, including CBT, DBT, and finding a therapist.",
        "links": [
            {"label": "APA - Finding a Therapist",
                "url": "https://www.apa.org/helpcenter/choose-therapist"}
        ]
    }
}

WELLNESS_TIPS = [
    "Take 3 deep breaths right now. Feel your shoulders relax 🌬️",
    "Drink a glass of water. Your brain needs hydration 💧", 
    "Write down 3 things you're grateful for today 🙏",
    "Stand up and stretch for 30 seconds 🤸",
    "Send a kind message to someone you care about 💝",
    "Look out a window and notice something beautiful in nature 🌿",
    "Put your phone away for 10 minutes and just be present 📱",
    "Smile at yourself in the mirror. You deserve kindness 😊"
]

def render_daily_tip():
    """Show a random wellness tip"""
    
    # Get a random tip
    if "current_tip" not in st.session_state:
        st.session_state.current_tip = random.choice(WELLNESS_TIPS)
    
    # Show the tip in a nice box
    st.info(st.session_state.current_tip)
    
    # Buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("💚 Helpful", key="tip_helpful"):
            st.success("Glad it helped! 😊")
    with col2:
        if st.button("🔄 New Tip", key="new_tip"):
            st.session_state.current_tip = random.choice(WELLNESS_TIPS)
            st.rerun()

def render_ambient_sounds():
    """Render calming music player in sidebar with soothing melodies"""
    st.markdown("""
    <div class="sidebar-music-section">
        <div class="music-header">🎵 Calming Sounds</div>
    </div>
    """, unsafe_allow_html=True)
    
    with st.expander("🎵 Choose Background Music", expanded=False):
        st.markdown("**Create a peaceful atmosphere:**")
        
        # YouTube calming music videos - peaceful instrumental and meditation music
        calming_music = {
            "🎹 Peaceful Piano": {
                "description": "Soft piano melodies for relaxation",
                "embed_id": "1ZYbU82GVz4",  # Beautiful relaxing piano music
                "duration": "3 hours"
            },
            "🧘 Meditation Music": {
                "description": "Gentle meditation and mindfulness music",
                "embed_id": "lFcSrYw-ARY",  # Relaxing meditation music
                "duration": "1 hour"
            },
            "🎻 Calm Instrumental": {
                "description": "Soothing instrumental music mix",
                "embed_id": "M4QVYDTmjEg",  # Beautiful instrumental music
                "duration": "2 hours"
            },
            "🌸 Zen Garden": {
                "description": "Peaceful zen music for inner calm",
                "embed_id": "5qap5aO4i9A",  # Zen music for relaxation
                "duration": "3 hours"
            },
            "💤 Sleep Music": {
                "description": "Ultra calming music for deep relaxation",
                "embed_id": "YQaW2gkV1iM",  # Sleep music, calming music
                "duration": "8 hours"
            },
            "🎶 Ambient Chillout": {
                "description": "Soft ambient music for stress relief",
                "embed_id": "rUxyKA_-grg",  # Chillout ambient music
                "duration": "1 hour"
            }
        }
        
        selected_music = st.selectbox(
            "Select calming music:",
            ["🔇 Silence"] + list(calming_music.keys()),
            key="ambient_sound_selector"
        )
        
        if selected_music != "🔇 Silence":
            music_data = calming_music[selected_music]
            
            st.markdown(f"**Now Playing: {selected_music}**")
            st.markdown(f"*{music_data['description']} ({music_data['duration']})*")
            
            # Embed YouTube video as audio player
            youtube_embed = f"""
            <div style="text-align: center; margin: 10px 0;">
                <iframe width="100%" height="80" 
                        src="https://www.youtube.com/embed/{music_data['embed_id']}?autoplay=0&loop=1&playlist={music_data['embed_id']}&controls=1&modestbranding=1&rel=0&showinfo=0" 
                        frameborder="0" 
                        allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" 
                        allowfullscreen>
                </iframe>
            </div>
            """
            
            st.markdown(youtube_embed, unsafe_allow_html=True)
            
            st.info("💡 **Tip**: Keep volume gentle (15-25%) for a peaceful atmosphere.")
            
            # Alternative: Direct links for manual opening
            st.markdown("---")
            st.markdown("**Alternative**: Open in new tab:")
            youtube_url = f"https://www.youtube.com/watch?v={music_data['embed_id']}"
            st.markdown(f"[🔗 Open {selected_music} on YouTube]({youtube_url})")
        
        else:
            st.info("🔇 Select music above to create a calming atmosphere.")
            st.markdown("---")
            st.markdown("**Benefits of calming music:**")
            st.markdown("• Reduces stress and anxiety naturally")
            st.markdown("• Promotes emotional well-being")  
            st.markdown("• Enhances mindfulness and focus")
            st.markdown("• Creates a therapeutic environment")

<<<<<<< HEAD
# Renders the sidebar with a pinned messages count and a button to navigate to the Pinned Messages page

def render_sidebar():
    """Renders the left sidebar with organized sections."""
    if "pinned_messages" in st.session_state and st.session_state.pinned_messages:
        pin_count = len(st.session_state.pinned_messages)
        st.markdown(f"""
        <div style="background-color: rgba(255,255,255,0.2); padding: 8px; border-radius: 8px; margin-bottom: 15px; text-align: center;">
            📌 <strong>{pin_count}</strong> pinned message{'s' if pin_count != 1 else ''}
        </div>
        """, unsafe_allow_html=True)
        

    # Pinned Messages navigation
    if st.button("📌 View Pinned Messages", use_container_width=True):
        st.session_state.active_page = "PinnedMessages"
        st.rerun()

=======
def render_sidebar():
    """Renders the left sidebar with organized sections."""
>>>>>>> 7aad9a9 (Clean commit: add project files, ignore env & checkpoints)
    
    with st.sidebar:
        # Theme Settings Section
        with st.expander("🎨 Appearance Settings", expanded=False):
            current_theme = get_current_theme()
            is_dark = current_theme["name"] == "Dark"

            # Palette selector (only for light mode)
            if not is_dark:
                palette_names = [p["name"] for p in PALETTES]
                selected_palette = st.selectbox(
                    "Choose a soothing color palette:",
                    palette_names,
                    index=palette_names.index(
                        st.session_state.get("palette_name", "Light")),
                    key="palette_selector",
                )
                if selected_palette != st.session_state.get("palette_name", "Light"):
                    set_palette(selected_palette)

            # Current theme display
            st.markdown(f"""
            <div class="theme-info-display">
                <strong>Current Theme:</strong><br>
                <span class="theme-name">{current_theme['name']} Mode</span>
            </div>
            """, unsafe_allow_html=True)
<<<<<<< HEAD
            
=======

>>>>>>> 7aad9a9 (Clean commit: add project files, ignore env & checkpoints)
            # Theme toggle button
            button_text = "🌙 Switch to Dark" if not is_dark else "☀️ Switch to Light"
            button_type = "primary" if not is_dark else "secondary"

            if st.button(
                button_text,
                key="sidebar_theme_toggle",
                use_container_width=True,
                type=button_type
            ):
                toggle_theme()

        render_profile_section()

        # Daily Wellness Tip
        render_daily_tip()
        
        # Ambient Sounds
        render_ambient_sounds()
        
        st.markdown("---")
        
        # Chat Management Section
        st.markdown("""
        <div class="sidebar-section-header">
            <h3>💬 Chat Sessions</h3>
            <p>Manage your AI conversations</p>
        </div>
        """, unsafe_allow_html=True)

        if "show_quick_start_prompts" not in st.session_state:
            st.session_state.show_quick_start_prompts = False
        if "pre_filled_chat_input" not in st.session_state:
            st.session_state.pre_filled_chat_input = ""
        if "send_chat_message" not in st.session_state:
            st.session_state.send_chat_message = False

        if st.button("➕ New Chat", key="new_chat", use_container_width=True, type="primary"):
            create_new_conversation()
            st.session_state.show_quick_start_prompts = True
            st.rerun()

        if st.session_state.show_quick_start_prompts:
            st.markdown("**💭 Quick Start Topics:**")
            quick_prompts = [
                "Feeling overwhelmed",
                "Need to vent about my day", 
                "How to manage stress?",
                "Tell me about anxiety"
            ]
            qp_cols = st.columns(2)
            for i, prompt in enumerate(quick_prompts):
                with qp_cols[i % 2]:
                    if st.button(f"✨ {prompt}", key=f"qp_{i}", use_container_width=True):
                        st.session_state.pre_filled_chat_input = prompt
                        st.session_state.send_chat_message = True
                        st.session_state.show_quick_start_prompts = False
                        st.rerun()

        # Conversation History
        if st.session_state.conversations:
            st.markdown("**📚 Recent Conversations:**")
            if "delete_candidate" not in st.session_state:
                for i, convo in enumerate(st.session_state.conversations):
                    is_active = i == st.session_state.active_conversation
                    button_style_icon = "🟢" if is_active else "📝"

                    col1, col2 = st.columns([5, 1])
                    with col1:
                        if st.button(
                            f"{button_style_icon} {convo['title'][:20]}...",
                            key=f"convo_{i}",
                            help=f"Started: {convo['date']}",
                            use_container_width=True
                        ):
                            st.session_state.active_conversation = i
                            st.rerun()
                    with col2:
                        if convo["messages"]:
                            if st.button("🗑️", key=f"delete_{i}", type="primary", use_container_width=True):
                                st.session_state.delete_candidate = i
                                st.rerun()
                        else:
                            st.button(
                                "🗑️",
                                key=f"delete_{i}",
                                type="primary",
                                use_container_width=True,
                                disabled=not convo["messages"]
                            )
            else:
                st.warning("⚠️ Are you sure you want to delete this conversation?")
                col_confirm, col_cancel = st.columns(2)

                if col_confirm.button("Yes, delete", key="confirm_delete"):
                    del st.session_state.conversations[st.session_state.delete_candidate]

                    from core.utils import save_conversations
                    save_conversations(st.session_state.conversations)

                    del st.session_state.delete_candidate
                    st.session_state.active_conversation = -1
                    st.rerun()

                if "cancel_clicked" not in st.session_state:
                    st.session_state.cancel_clicked = False

                if col_cancel.button("Cancel", key="cancel_delete"):
                    if not st.session_state.cancel_clicked:
                        st.session_state.cancel_clicked = True
                        del st.session_state.delete_candidate
                        st.rerun()
                else:
                    st.session_state.cancel_clicked = False
        else:
<<<<<<< HEAD
            st.info("No conversations yet. Start a new chat!")
=======
            st.info("No conversations yet. Start a new chat!")
>>>>>>> 7aad9a9 (Clean commit: add project files, ignore env & checkpoints)
