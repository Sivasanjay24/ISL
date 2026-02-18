import cv2
import mediapipe as mp
import torch
import torch.nn as nn
import pickle
import numpy as np
import time
import os
import tkinter as tk
from tkinter import ttk
from deep_translator import GoogleTranslator
from PIL import Image, ImageFont, ImageDraw

# --- Configuration ---
MODEL_PATH = 'landmark_model.pth'
LABEL_MAP_PATH = 'label_map.pkl'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CONFIDENCE_THRESHOLD = 0.90
REQUIRED_STABILITY_TIME = 1.2

# --- Font Mapping (Language Code -> Font Filename) ---
# Ensure these .ttf files are in your project folder!
FONT_FILES = {
    'ta': 'tamil.ttf',
    'te': 'telugu.ttf',
    'kn': 'kannada.ttf',
    'ml': 'malayalam.ttf',
    'hi': 'hindi.ttf',
    'en': 'arial.ttf' # Fallback
}

# --- 1. Language Selection Popup (GUI) ---
selected_lang_code = 'ta' # Default
selected_lang_name = 'Tamil'

def launch_gui():
    global selected_lang_code, selected_lang_name
    
    root = tk.Tk()
    root.title("ISL System Setup")
    root.geometry("300x180")

    tk.Label(root, text="Select Output Language:", font=("Arial", 12, "bold")).pack(pady=10)

    languages = {
        'Tamil': 'ta',
        'Hindi': 'hi',
        'Telugu': 'te',
        'Kannada': 'kn',
        'Malayalam': 'ml'
    }

    combo = ttk.Combobox(root, values=list(languages.keys()), state="readonly", font=("Arial", 10))
    combo.current(0)
    combo.pack(pady=5)

    def on_start():
        global selected_lang_code, selected_lang_name
        selected_lang_name = combo.get()
        selected_lang_code = languages[selected_lang_name]
        root.destroy()

    tk.Button(root, text="Start Camera", command=on_start, bg="green", fg="white", font=("Arial", 11)).pack(pady=20)
    root.mainloop()

# Launch GUI
launch_gui()
print(f"System Starting... Target: {selected_lang_name} ({selected_lang_code})")

# --- 2. Model Architecture ---
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

# --- 3. Load Resources ---
try:
    with open(LABEL_MAP_PATH, 'rb') as f:
        label_map = pickle.load(f)
    
    # Input size 84 = 21 landmarks * 2 coords * 2 hands
    model = LandmarkModel(input_size=84, num_classes=len(label_map))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
except Exception as e:
    print(f"CRITICAL ERROR: {e}")
    print("Ensure 'landmark_model.pth' and 'label_map.pkl' are present.")
    exit()

# --- Initialize MediaPipe ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# --- Helper: Draw Non-English Text ---
def put_text_pil(img, text, position, font_filename, size=28, color=(255, 255, 0)):
    # Convert to PIL format
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    # Construct font path
    font_path = os.path.join(os.path.dirname(__file__), font_filename)
    
    try:
        font = ImageFont.truetype(font_path, size)
    except IOError:
        # Fallback if specific font not found
        print(f"WARNING: Font '{font_filename}' not found. Using default.")
        font = ImageFont.load_default()
    
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# --- Variables ---
sentence = ""
translated_sentence = ""
stable_letter = None
start_time = None
last_action_time = 0

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)
    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    current_char = None
    confidence = 0
    landmarks_data = []

    # --- Hand Processing & Feature Extraction ---
    if results.multi_hand_landmarks:
        for hand_idx in range(2):
            if hand_idx < len(results.multi_hand_landmarks):
                hand_landmarks = results.multi_hand_landmarks[hand_idx]
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                for lm in hand_landmarks.landmark:
                    landmarks_data.extend([lm.x, lm.y])
            else:
                landmarks_data.extend([0.0] * 42) # Pad missing hand

        if len(landmarks_data) == 84:
            input_tensor = torch.tensor(landmarks_data, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                outputs = model(input_tensor)
                probs = torch.softmax(outputs, dim=1)
                conf, pred_idx = torch.max(probs, 1)
                
                if conf.item() > CONFIDENCE_THRESHOLD:
                    current_char = label_map.get(pred_idx.item())
                    confidence = conf.item()

    # --- Sentence Logic ---
    if current_char:
        if current_char == stable_letter:
            if start_time is None: start_time = time.time()
            elapsed = time.time() - start_time
            
            # Progress Bar Logic
            is_cmd = current_char in ["SPACE", "DELETE", "TRANSLATE"]
            bar_color = (0, 0, 255) if is_cmd else (0, 255, 0) # Red for Cmd, Green for Text
            
            cv2.rectangle(frame, (0, H-10), (int((elapsed/REQUIRED_STABILITY_TIME)*W), H), bar_color, -1)
            
            if elapsed > REQUIRED_STABILITY_TIME:
                # Debounce: prevent rapid-fire triggering
                if time.time() - last_action_time > 1.0:
                    
                    if current_char == "SPACE":
                        sentence += " "
                        last_action_time = time.time()
                        
                    elif current_char == "DELETE":
                        sentence = sentence[:-1]
                        translated_sentence = "" # Reset translation
                        last_action_time = time.time()
                        
                    elif current_char == "TRANSLATE":
                        if sentence.strip():
                            print(f"Translating '{sentence}' to {selected_lang_name}...")
                            try:
                                # Convert to lowercase to fix "ICE" -> "ice" issue
                                text_to_translate = sentence.lower()
                                
                                # Perform Translation
                                t = GoogleTranslator(source='en', target=selected_lang_code).translate(text_to_translate)
                                translated_sentence = t
                            except Exception as e:
                                print(f"Translation Error: {e}")
                                translated_sentence = "Error"
                        last_action_time = time.time()
                        
                    elif not is_cmd:
                        # Normal Character Logic
                        if not sentence.endswith(current_char):
                            sentence += current_char
                            last_action_time = time.time()
                
                start_time = None # Reset timer
        else:
            stable_letter = current_char
            start_time = time.time()
    else:
        stable_letter = None
        start_time = None

    # --- UI Rendering ---
    # Bottom Info Panel
    cv2.rectangle(frame, (0, H - 120), (W, H), (0, 0, 0), -1)
    
    # 1. English Text
    cv2.putText(frame, f"ENG: {sentence}", (20, H - 80), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # 2. Translated Text (Dynamic Font)
    if translated_sentence:
        display_text = f"{selected_lang_name}: {translated_sentence}"
        
        # Select the correct font file based on language code
        font_file = FONT_FILES.get(selected_lang_code, "arial.ttf")
        
        frame = put_text_pil(frame, display_text, (20, H - 40), font_file, size=32, color=(0, 255, 255))

    # 3. Live Prediction Overlay
    if current_char:
        cv2.putText(frame, f"Sign: {current_char}", (W - 250, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('ISL Pro System', frame)

    # Keyboard Shortcuts
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break
    elif key == ord('c'): # Clear All
        sentence = ""
        translated_sentence = ""

cap.release()
cv2.destroyAllWindows()