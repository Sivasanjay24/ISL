import streamlit as st
import cv2
import mediapipe as mp
import torch
import torch.nn as nn
import pickle
import numpy as np
import time
import os
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
from deep_translator import GoogleTranslator
from PIL import Image, ImageFont, ImageDraw

# --- Streamlit Page Config ---
st.set_page_config(page_title="ISL Web Interpreter", layout="wide")

# --- Configuration ---
MODEL_PATH = 'landmark_model.pth'
LABEL_MAP_PATH = 'label_map.pkl'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CONFIDENCE_THRESHOLD = 0.90
REQUIRED_STABILITY_TIME = 1.2

FONT_FILES = {
    'ta': 'tamil.ttf',
    'te': 'telugu.ttf',
    'kn': 'kannada.ttf',
    'ml': 'malayalam.ttf',
    'hi': 'hindi.ttf',
    'en': 'arial.ttf' 
}

# --- Model Architecture ---
class LandmarkModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LandmarkModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    def forward(self, x):
        return self.network(x)

# --- Load Resources (Cached) ---
@st.cache_resource
def load_model_and_labels():
    try:
        with open(LABEL_MAP_PATH, 'rb') as f:
            label_map = pickle.load(f)
        
        model = LandmarkModel(input_size=84, num_classes=len(label_map))
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        return model, label_map
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None, None

model, label_map = load_model_and_labels()

# --- Helper: Draw Non-English Text ---
def put_text_pil(img, text, position, font_filename, size=28, color=(255, 255, 0)):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    font_path = os.path.join(os.path.dirname(__file__), font_filename)
    try:
        font = ImageFont.truetype(font_path, size)
    except IOError:
        font = ImageFont.load_default()
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# --- WebRTC Video Processor ---
class ISLProcessor:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
        self.mp_drawing = mp.solutions.drawing_utils
        
        self.sentence = ""
        self.translated_sentence = ""
        self.stable_letter = None
        self.start_time = None
        self.last_action_time = 0
        
        self.selected_lang_code = 'ta'
        self.selected_lang_name = 'Tamil'

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        H, W, _ = img.shape
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        results = self.hands.process(img_rgb)
        current_char = None
        confidence = 0
        landmarks_data = []

        # Feature Extraction
        if results.multi_hand_landmarks:
            for hand_idx in range(2):
                if hand_idx < len(results.multi_hand_landmarks):
                    hand_landmarks = results.multi_hand_landmarks[hand_idx]
                    self.mp_drawing.draw_landmarks(img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    for lm in hand_landmarks.landmark:
                        landmarks_data.extend([lm.x, lm.y])
                else:
                    landmarks_data.extend([0.0] * 42)

            # Model Prediction
            if len(landmarks_data) == 84 and model is not None:
                input_tensor = torch.tensor(landmarks_data, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    outputs = model(input_tensor)
                    probs = torch.softmax(outputs, dim=1)
                    conf, pred_idx = torch.max(probs, 1)
                    
                    if conf.item() > CONFIDENCE_THRESHOLD:
                        current_char = label_map.get(pred_idx.item())
                        confidence = conf.item()

        # Logic & Stability Filter
        if current_char:
            if current_char == self.stable_letter:
                if self.start_time is None: self.start_time = time.time()
                elapsed = time.time() - self.start_time
                
                is_cmd = current_char in ["SPACE", "DELETE", "TRANSLATE"]
                bar_color = (0, 0, 255) if is_cmd else (0, 255, 0)
                
                cv2.rectangle(img, (0, H-10), (int((elapsed/REQUIRED_STABILITY_TIME)*W), H), bar_color, -1)
                
                if elapsed > REQUIRED_STABILITY_TIME:
                    if time.time() - self.last_action_time > 1.0:
                        if current_char == "SPACE":
                            self.sentence += " "
                        elif current_char == "DELETE":
                            self.sentence = self.sentence[:-1]
                            self.translated_sentence = ""
                        elif current_char == "TRANSLATE":
                            if self.sentence.strip():
                                try:
                                    t = GoogleTranslator(source='en', target=self.selected_lang_code).translate(self.sentence.lower())
                                    self.translated_sentence = t
                                except Exception:
                                    self.translated_sentence = "Error"
                        elif not is_cmd:
                            if not self.sentence.endswith(current_char):
                                self.sentence += current_char
                        
                        self.last_action_time = time.time()
                    self.start_time = None
            else:
                self.stable_letter = current_char
                self.start_time = time.time()
        else:
            self.stable_letter = None
            self.start_time = None

        # UI Rendering
        cv2.rectangle(img, (0, H - 120), (W, H), (0, 0, 0), -1)
        cv2.putText(img, f"ENG: {self.sentence}", (20, H - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        if self.translated_sentence:
            display_text = f"{self.selected_lang_name}: {self.translated_sentence}"
            font_file = FONT_FILES.get(self.selected_lang_code, "arial.ttf")
            img = put_text_pil(img, display_text, (20, H - 40), font_file, size=32, color=(0, 255, 255))

        if current_char:
            cv2.putText(img, f"Sign: {current_char}", (W - 250, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- Streamlit Frontend UI ---
st.title("Indian Sign Language Web Interpreter")

col1, col2 = st.columns([1, 3])

with col1:
    st.markdown("### Settings")
    languages = {'Tamil': 'ta', 'Hindi': 'hi', 'Telugu': 'te', 'Kannada': 'kn', 'Malayalam': 'ml'}
    selected_name = st.selectbox("Output Language", list(languages.keys()))
    selected_code = languages[selected_name]

    st.markdown("### Performance")
    quality = st.radio("Video Quality", ["Low (Best for Mobile)", "Standard (Balanced)", "High (Desktop Only)"], index=1)

with col2:
    # Twilio TURN Server Integration
    @st.cache_data
    def get_ice_servers():
        try:
            account_sid = st.secrets["TWILIO_ACCOUNT_SID"]
            auth_token = st.secrets["TWILIO_AUTH_TOKEN"]
            from twilio.rest import Client
            client = Client(account_sid, auth_token)
            token = client.tokens.create()
            return token.ice_servers
        except Exception as e:
            return [{"urls": ["stun:stun.l.google.com:19302"]}]

    rtc_config = RTCConfiguration({"iceServers": get_ice_servers()})

    # Dynamic Camera Constraints based on User Device
    if "Low" in quality:
        vid_constraints = {"width": {"ideal": 320}, "height": {"ideal": 240}, "frameRate": {"ideal": 15, "max": 15}}
    elif "Standard" in quality:
        vid_constraints = {"width": {"ideal": 640}, "height": {"ideal": 480}, "frameRate": {"ideal": 20, "max": 24}}
    else:
        vid_constraints = {"width": {"ideal": 1280}, "height": {"ideal": 720}, "frameRate": {"ideal": 30}}

    ctx = webrtc_streamer(
        key="isl-streamer",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=rtc_config,
        video_processor_factory=ISLProcessor,
        media_stream_constraints={
            "video": vid_constraints,
            "audio": False
        },
        async_processing=True
    )

    # State Sync
    if ctx.video_processor:
        ctx.video_processor.selected_lang_name = selected_name
        ctx.video_processor.selected_lang_code = selected_code
        
        if st.button("Clear Text"):
            ctx.video_processor.sentence = ""
            ctx.video_processor.translated_sentence = ""