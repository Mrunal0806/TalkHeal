import streamlit as st
import random
import json
import re
from streamlit.components.v1 import html as st_html


# --- Template-Based Generation System ---
# All categories now use templates to generate a wide variety of prompts.
CARD_TEMPLATES = {
    "Grounding": {
        "templates": [
            {"title": "Object Focus", "text": "Find something {color} in the room. Describe three details about it to yourself."},
            {"title": "Texture Hunt", "text": "Touch something nearby that feels {texture}. Focus on that sensation for a moment."},
            {"title": "Sound Separation", "text": "Close your eyes and try to identify the most {sound_type} sound you can hear right now."},
            {"title": "5-4-3-2-1 Senses", "text": "Name 5 things you see, 4 things you feel, 3 things you hear, 2 things you smell, and 1 thing you can taste."},
            {"title": "Body Scan", "text": "Press your {body_part} firmly against the {surface} you're on. Feel the solid connection to the ground."},
        ],
        "fills": {
            "color": ["blue", "red", "green", "yellow", "black", "white", "orange", "purple"],
            "texture": ["smooth", "rough", "soft", "hard", "cool", "warm"],
            "sound_type": ["distant", "close", "quiet", "loud", "rhythmic"],
            "body_part": ["feet", "hands", "back"],
            "surface": ["floor", "chair", "wall", "desk"],
        }
    },
    "Mindfulness": {
        "templates": [
            {"title": "Mindful Observation", "text": "Take a moment to mindfully observe the {noun} near you. Notice its color, texture, and shape without judgment."},
            {"title": "Mindful Sip", "text": "Slowly sip a {drink} and notice everything about it: the temperature, the taste, and the feeling in your mouth."},
            {"title": "Sensory Focus", "text": "Focus on the sensation of your {body_part} touching the {surface} you're on. Notice the pressure and texture."},
            {"title": "Three Mindful Breaths", "text": "Take three slow, deep breaths. Focus completely on the sensation of the breath entering and leaving your body."},
            {"title": "Cloud Gazing", "text": "If you can see the sky, watch the clouds for a minute. If not, imagine them in your mind's eye."},
        ],
        "fills": {
            "noun": ["pen", "plant", "light in the room", "cup", "window", "keyboard"],
            "drink": ["glass of water", "cup of tea", "warm drink", "cool beverage"],
            "body_part": ["feet", "hands", "back", "legs"],
            "surface": ["floor", "chair", "desk", "wall"],
        }
    },
    "Movement": {
        "templates": [
            {"title": "Gentle Stretch", "text": "Gently stretch your {body_part}. Hold for 15 seconds, breathing slowly."},
            {"title": "Quick Shake-out", "text": "Gently shake your {body_part} for a few moments to release any stored tension."},
            {"title": "Short Walk", "text": "If you can, take a brief walk, even just to the {place} and back. Focus on your steps."},
            {"title": "Joint Roll", "text": "Slowly and gently roll your {joint} in a circle a few times in each direction."},
            {"title": "Posture Check", "text": "Sit or stand up straight. Gently pull your shoulders back and lengthen your spine."},
        ],
        "fills": {
            "body_part": ["neck", "shoulders", "arms", "wrists", "legs", "back", "fingers"],
            "place": ["kitchen", "window", "door", "end of the hallway"],
            "joint": ["ankles", "wrists", "shoulders", "neck"],
        }
    },
    "Reflection": {
        "templates": [
            {"title": "Gratitude Moment", "text": "What is one small thing about your {time_period} that you are grateful for?"},
            {"title": "One-Word Check-in", "text": "If you had to describe your current feeling in just one word, what would it be? Acknowledge it without judgment."},
            {"title": "Small Win", "text": "Think of one thing, no matter how small, that you've accomplished today. Give yourself credit for it."},
            {"title": "Future Self", "text": "What is one small, kind thing you can do right now that your future self will thank you for?"},
            {"title": "Letting Go", "text": "Is there a small worry from today that you can acknowledge and mentally place in a box to deal with later?"},
        ],
        "fills": {
            "time_period": ["day", "morning", "last hour", "current space"],
        }
    },
    "Affirmation": {
        "templates": [
            {"title": "Permission Slip", "text": "Give yourself permission to feel exactly as you do right now. Say to yourself, 'It's okay to feel {emotion}.'"},
            {"title": "Statement of Strength", "text": "Remind yourself of your resilience. Say, 'I have handled {challenge} before, and I can handle this.'"},
            {"title": "Simple Truth", "text": "Repeat this simple phrase to yourself: 'I am doing the best I can with what I have right now.'"},
            {"title": "Future Focus", "text": "Acknowledge that feelings are temporary. Say, 'This feeling of {emotion} is not permanent.'"},
            {"title": "Self-Acceptance", "text": "Look in a mirror or close your eyes and say, 'I accept myself, flaws and all.'"},
        ],
        "fills": {
            "emotion": ["overwhelmed", "anxious", "sad", "tired", "confused"],
            "challenge": ["difficult things", "uncertainty", "stress"],
        }
    },
    "Creative": {
        "templates": [
            {"title": "Quick Doodle", "text": "Grab a pen and paper and spend one minute doodling a {shape} or a {creature}."},
            {"title": "Creative Description", "text": "Look at a {object} nearby. Think of three unusual ways to describe it."},
            {"title": "Story Starter", "text": "Imagine a tiny character who lives in a {place}. What is the first thing they do today?"},
            {"title": "Six-Word Story", "text": "Try to write a six-word story about the feeling of {emotion}."},
            {"title": "Color Association", "text": "What does the color {color} feel like? Write down three words that come to mind."},
        ],
        "fills": {
            "shape": ["spiral", "series of circles", "zigzag pattern", "checkerboard"],
            "creature": ["friendly monster", "tiny dragon", "robot animal"],
            "object": ["lamp", "book", "chair", "shoe"],
            "place": ["matchbox", "houseplant", "teacup"],
            "emotion": ["calm", "joy", "surprise", "longing"],
            "color": ["blue", "green", "yellow", "purple"],
        }
    }
}

# All cards are now generated from templates.
STATIC_CARDS = []


def _generate_card(category):
    """Generates a new, unique card from the template library for a given category."""
    if category not in CARD_TEMPLATES:
        return None

    spec = CARD_TEMPLATES[category]
    template = random.choice(spec["templates"])
    
    # Find all placeholders like {noun} in the text
    placeholders = re.findall(r"\{(\w+)\}", template["text"])
    
    # Create a dictionary to hold the replacements
    replacements = {}
    for p in placeholders:
        if p in spec["fills"]:
            replacements[p] = random.choice(spec["fills"][p])

    # Format the text and title
    generated_text = template["text"].format(**replacements)
    
    return {
        "title": template["title"],
        "text": generated_text,
        "category": category,
        "time": "1-3 min", # Generic time for generated cards
        "generated": True # Flag to identify generated cards
    }

def _get_all_cards():
    """Returns a list of all possible cards, both static and one example from each template."""
    # This is mainly for populating the category list correctly
    all_cards = STATIC_CARDS
    for category, spec in CARD_TEMPLATES.items():
        # Add a representative card for the category to exist
        all_cards.append({"category": category})
    return all_cards


def _ensure_state():
    if "quick_coping_favorites" not in st.session_state:
        st.session_state.quick_coping_favorites = []
    if "quick_coping_current" not in st.session_state:
        # Initialize with a random card from any category
        initial_category = random.choice(list(CARD_TEMPLATES.keys()))
        st.session_state.quick_coping_current = _generate_card(initial_category)
    if "quick_coping_category" not in st.session_state:
        st.session_state.quick_coping_category = "All"
    if "quick_coping_last_card" not in st.session_state:
        st.session_state.quick_coping_last_card = None


def _get_random_candidate(category):
    """
    Gets a new card. If a category is specified, it generates from that template.
    For "All", it picks a random category first, then generates.
    It also tries not to return the exact same card twice in a row.
    """
    target_category = category
    if category == "All":
        target_category = random.choice(list(CARD_TEMPLATES.keys()))

    # Try a few times to get a card that's different from the last one
    for _ in range(5): # Max 5 tries to prevent infinite loops
        candidate = _generate_card(target_category)
        if candidate != st.session_state.get("quick_coping_last_card"):
            st.session_state.quick_coping_last_card = candidate
            return candidate
    
    # If it fails after 5 tries, just return the last one it found
    return candidate


def render_quick_coping_cards():
    """Render the Quick Coping Cards widget with richer cards and fallback behavior."""
    _ensure_state()

    st.markdown("### 🃏 Quick Coping Cards")
    st.markdown("_Instant, friendly strategies you can try in moments of distress or overwhelm._")

    all_possible_cards = _get_all_cards()
    categories = sorted({c["category"] for c in all_possible_cards})
    categories.insert(0, "All")

    ICONS = {
        "All": "🃏", "Grounding": "🌿", "Movement": "🏃", "Reflection": "📝",
        "Affirmation": "💪", "Creative": "🎨", "Mindfulness": "🧘",
    }

    left, right = st.columns([3, 1])
    with left:
        labels = [f"{ICONS.get(cat,'')}  {cat}" for cat in categories]
        sel_index = categories.index(st.session_state.quick_coping_category) if st.session_state.quick_coping_category in categories else 0
        chosen_label = st.selectbox("Category", labels, index=sel_index, key="quick_coping_category_select")
        selected_category = chosen_label.split()[-1]
        
        # If category changed, get a new card immediately
        if selected_category != st.session_state.quick_coping_category:
            st.session_state.quick_coping_category = selected_category
            st.session_state.quick_coping_current = _get_random_candidate(selected_category)
            st.rerun()

    with right:
        st.write("") # for alignment
        st.write("") # for alignment
        if st.button("🎲 New card", key="quick_coping_new", use_container_width=True):
            st.session_state.quick_coping_current = _get_random_candidate(st.session_state.quick_coping_category)
            st.rerun()

    # Render the selected card
    card = st.session_state.quick_coping_current
    
    if card:
        with st.container():
            icon = ICONS.get(card.get("category"), "🃏")
            title = card.get("title") or card.get("text")
            text = card.get("text")
            time_est = card.get("time")
            steps = card.get("steps", [])

            time_html = f"⏱ {time_est}" if time_est else ""
            if steps:
                steps_html = '<ol style="margin-left:18px;color:#374151;">' + ''.join([f'<li style="margin-bottom:6px;font-size:14px;">{s}</li>' for s in steps]) + '</ol>'
            else:
                steps_html = ''

            card_html = f"""
            <div style="display:flex;justify-content:center;margin:20px 0;">
                <div style="
                    width:420px;
                    height:450px;
                    border-radius:20px;
                    padding:0;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    box-shadow: 0 20px 40px rgba(102, 126, 234, 0.3);
                    display:flex;
                    flex-direction:column;
                    font-family: 'Inter', system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
                    transition: transform 0.3s ease-in-out, box-shadow 0.3s ease-in-out;
                    position: relative;
                    overflow: hidden;
                "
                onmouseover="this.style.transform='scale(1.03) rotate(1deg)'; this.style.boxShadow='0 25px 50px rgba(102, 126, 234, 0.4)';"
                onmouseout="this.style.transform='scale(1) rotate(0deg)'; this.style.boxShadow='0 20px 40px rgba(102, 126, 234, 0.3)';">
                    
                    <!-- Decorative background pattern -->
                    <div style="position: absolute; top: -50px; right: -50px; width: 150px; height: 150px; background: rgba(255,255,255,0.1); border-radius: 50%; z-index: 1;"></div>
                    <div style="position: absolute; bottom: -30px; left: -30px; width: 100px; height: 100px; background: rgba(255,255,255,0.05); border-radius: 50%; z-index: 1;"></div>
                    
                    <!-- Card Header -->
                    <div style="background: rgba(255,255,255,0.15); padding: 20px; border-radius: 20px 20px 0 0; backdrop-filter: blur(10px); position: relative; z-index: 2;">
                        <div style="display:flex;align-items:center;gap:15px;margin-bottom:10px;">
                            <div style="width:60px;height:60px;border-radius:16px;background:rgba(255,255,255,0.9);display:flex;align-items:center;justify-content:center;font-size:28px;box-shadow: 0 8px 16px rgba(0,0,0,0.1);">{icon}</div>
                            <div style="flex:1;">
                                <div style="font-size:22px;color:#ffffff;font-weight:700;text-shadow: 0 2px 4px rgba(0,0,0,0.3);line-height:1.2;">{title}</div>
                                <div style="font-size:14px;color:rgba(255,255,255,0.9);margin-top:4px;font-weight:500;">{card.get('category','General')} Strategy</div>
                            </div>
                            <div style="background:rgba(255,255,255,0.2);padding:8px 16px;border-radius:20px;backdrop-filter: blur(5px);">
                                <div style="font-size:12px;color:#ffffff;font-weight:600;">{time_html}</div>
                            </div>
                        </div>
                        
                        <!-- Progress indicator -->
                        <div style="background:rgba(255,255,255,0.2);height:4px;border-radius:2px;overflow:hidden;">
                            <div style="background:#ffffff;height:100%;width:75%;border-radius:2px;"></div>
                        </div>
                    </div>

                    <!-- Card Body -->
                    <div style="flex:1;padding:25px;background:#ffffff;position:relative;z-index:2;">
                        <div style="background:linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);border-radius:12px;padding:20px;margin-bottom:15px;border-left:4px solid #667eea;">
                            <div style="font-size:17px;color:#1e293b;line-height:1.7;font-weight:500;">{text}</div>
                        </div>
                        
                        {steps_html if steps else '<div style="text-align:center;padding:15px;color:#64748b;font-style:italic;">Take a deep breath and begin when you are ready</div>'}
                        
                        <!-- Motivational quote -->
                        <div style="background:rgba(102, 126, 234, 0.05);border-radius:8px;padding:15px;margin-top:15px;border:1px solid rgba(102, 126, 234, 0.1);">
                            <div style="font-size:14px;color:#475569;text-align:center;font-style:italic;">"Small steps lead to big changes. You've got this! 💪"</div>
                        </div>
                    </div>

                    <!-- Card Footer -->
                    <div style="background:linear-gradient(135deg, #f1f5f9 0%, #e2e8f0 100%);padding:15px 25px;border-radius:0 0 20px 20px;position:relative;z-index:2;">
                        <div style="display:flex;justify-content:space-between;align-items:center;">
                            <div style="font-size:13px;color:#64748b;font-weight:500;">
                                <span style="color:#22c55e;">●</span> Ready to try
                            </div>
                            <div style="font-size:12px;color:#64748b;background:#ffffff;padding:4px 8px;border-radius:12px;font-weight:600;">
                                #{card.get('category','General').upper()}
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            """
            st_html(card_html, height=400)
    else:
        st.info("No coping cards available right now. Try 'Surprise me' or switch categories.")

    if card:
        # Action buttons container
        with st.container():
            a, b = st.columns([1, 1])
            with a:
                if st.button("⭐ Save", key="quick_coping_fav", use_container_width=True):
                    # Use a tuple of title and text for uniqueness check
                    card_id = (card.get('title'), card.get('text'))
                    is_in_favorites = any(
                        (fav.get('title'), fav.get('text')) == card_id
                        for fav in st.session_state.quick_coping_favorites
                    )
                    if not is_in_favorites:
                        st.session_state.quick_coping_favorites.append(card)
                        st.success("Saved to favorites (session)")
                        st.rerun()
                    else:
                        st.info("Already in favorites")
            with b:
                if st.button("💬 Use in chat", key="quick_coping_use", use_container_width=True):
                    st.session_state.pre_filled_chat_input = f"Trying a coping strategy: {card.get('text')}"
                    st.session_state.send_chat_message = True
                    st.success("Prepared message for chat")
                    st.rerun()

        st.markdown("---")

        # Favorites
        with st.expander("⭐ My Favorite Coping Strategies", expanded=bool(st.session_state.quick_coping_favorites)):
            if st.session_state.quick_coping_favorites:
                for i, fav in enumerate(st.session_state.quick_coping_favorites):
                    st.markdown(f"**{i+1}.** {fav.get('title', fav.get('text'))}  _(Category: {fav.get('category')})_")
                    use_col, rem_col = st.columns([1, 1])
                    with use_col:
                        if st.button("Use", key=f"use_fav_{i}", use_container_width=True):
                            st.session_state.quick_coping_current = fav
                            st.rerun()
                    with rem_col:
                        if st.button("Remove", key=f"remove_fav_{i}", use_container_width=True):
                            st.session_state.quick_coping_favorites.pop(i)
                            st.rerun()
                if st.button("🗑️ Clear all favorites", key="quick_coping_clear_favs", use_container_width=True):
                    st.session_state.quick_coping_favorites = []
                    st.rerun()
            else:
                st.write("No favorites yet — save helpful cards for quick access.")
