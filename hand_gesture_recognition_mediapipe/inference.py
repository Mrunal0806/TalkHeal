import cv2
import copy
import numpy as np
import streamlit as st
from collections import Counter, deque
import mediapipe as mp
import os, csv

# Import models and utils
from hand_gesture_recognition_mediapipe.model import KeyPointClassifier, PointHistoryClassifier
from hand_gesture_recognition_mediapipe.utils import CvFpsCalc
from hand_gesture_recognition_mediapipe.app import (
    calc_bounding_rect,
    calc_landmark_list,
    pre_process_landmark,
    pre_process_point_history,
    draw_landmarks,
    draw_bounding_rect,
    draw_info_text,
    draw_point_history,
    draw_info,
)

# -------------------- Setup --------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
)

keypoint_classifier = KeyPointClassifier()
point_history_classifier = PointHistoryClassifier()

# Load labels
base_dir = os.path.dirname(__file__)
with open(os.path.join(base_dir, "model/keypoint_classifier/keypoint_classifier_label.csv"), encoding="utf-8-sig") as f:
    keypoint_labels = [row[1] for row in csv.reader(f)]
with open(os.path.join(base_dir, "model/point_history_classifier/point_history_classifier_label.csv"), encoding="utf-8-sig") as f:
    point_history_labels = [row[0] for row in csv.reader(f)]


# -------------------- Gesture Mode --------------------
def gesture_mode():
    st.title("🖐 Gesture Input Mode")

    if "gesture_active" not in st.session_state:
        st.session_state.gesture_active = False

    col1, col2 = st.columns(2)
    with col1:
        start_button = st.button("▶️ Start Gesture Mode", disabled=st.session_state.gesture_active)
    with col2:
        stop_button = st.button("⏹ Stop Gesture Mode", disabled=not st.session_state.gesture_active)

    FRAME_WINDOW = st.image([])  # placeholder for webcam

    if start_button:
        st.session_state.gesture_active = True
    if stop_button:
        st.session_state.gesture_active = False

    if not st.session_state.gesture_active:
        st.info("Click **Start Gesture Mode** to activate webcam.")
        return

    # Open camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

    cvFps = CvFpsCalc(buffer_len=10)
    history_len = 16
    point_history = deque(maxlen=history_len)
    finger_history = deque(maxlen=history_len)

    detected_text = None

    while st.session_state.gesture_active:
        fps = cvFps.get()
        ret, frame = cap.read()
        if not ret:
            st.warning("⚠️ Unable to access webcam.")
            break

        frame = cv2.flip(frame, 1)
        debug = copy.deepcopy(frame)

        # Process hands
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        res = hands.process(rgb)
        rgb.flags.writeable = True

        detected_text = None

        if res.multi_hand_landmarks:
            for lm, handedness in zip(res.multi_hand_landmarks, res.multi_handedness):
                brect = calc_bounding_rect(debug, lm)
                landmark_list = calc_landmark_list(debug, lm)

                pp_landmarks = pre_process_landmark(landmark_list)
                pp_point_history = pre_process_point_history(debug, point_history)

                # Classify gesture
                sign_id = keypoint_classifier(pp_landmarks)

                if sign_id == 2:  # Index finger pointing
                    point_history.append(landmark_list[8])
                else:
                    point_history.append([0, 0])

                # Classify motion gesture
                fg_id = 0
                if len(pp_point_history) == history_len * 2:
                    fg_id = point_history_classifier(pp_point_history)
                finger_history.append(fg_id)
                most_common = Counter(finger_history).most_common(1)[0][0]

                detected_text = keypoint_labels[sign_id]

                # Draw debug info
                debug = draw_bounding_rect(True, debug, brect)
                debug = draw_landmarks(debug, landmark_list)
                debug = draw_info_text(debug, brect, handedness,
                                       keypoint_labels[sign_id],
                                       point_history_labels[most_common])
        else:
            point_history.append([0, 0])

        debug = draw_point_history(debug, point_history)
        debug = draw_info(debug, fps, 0, -1)

        # Show detected gesture
        if detected_text:
            st.markdown(f"### ✋ Detected Gesture: **{detected_text}**")

        FRAME_WINDOW.image(cv2.cvtColor(debug, cv2.COLOR_BGR2RGB))

        # Yield control back to Streamlit to keep UI responsive
        if not st.session_state.gesture_active:
            break

    cap.release()
    st.success("✅ Gesture mode stopped.")
