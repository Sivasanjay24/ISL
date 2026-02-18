import cv2
import mediapipe as mp
import csv
import os
import time

DATA_FILE = 'hand_sign_data.csv'
MAX_HANDS = 2  
LANDMARKS_PER_HAND = 21
COORDINATES_PER_LANDMARK = 2 
TOTAL_FEATURES = MAX_HANDS * LANDMARKS_PER_HAND * COORDINATES_PER_LANDMARK 

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=MAX_HANDS,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)


sign_name = input("Enter the name of the sign (label): ").upper()
num_samples = int(input("Enter the number of samples to collect: "))

cap = cv2.VideoCapture(0)

print(f"\nCollecting data for: {sign_name}")
print("Position your hand(s) in the frame.")
print("Press 'S' to start capturing...")


while True:
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)
    
    cv2.putText(frame, "READY? PRESS 'S' TO START", (80, 240), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
    cv2.imshow('Data Collection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('s'):
        break


collected_count = 0
while collected_count < num_samples:
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)
    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    landmarks_data = []

    if results.multi_hand_landmarks:
        for hand_idx in range(MAX_HANDS):
            if hand_idx < len(results.multi_hand_landmarks):
                hand_landmarks = results.multi_hand_landmarks[hand_idx]
                for lm in hand_landmarks.landmark:
                    landmarks_data.extend([lm.x, lm.y])
                
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            else:
                landmarks_data.extend([0.0] * (LANDMARKS_PER_HAND * COORDINATES_PER_LANDMARK))

        with open(DATA_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([sign_name] + landmarks_data)
        
        collected_count += 1
    else:
        cv2.putText(frame, "NO HAND DETECTED", (150, 240), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.rectangle(frame, (0, 0), (280, 60), (0, 0, 0), -1)
    cv2.putText(frame, f"Samples: {collected_count}/{num_samples}", (10, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    cv2.imshow('Data Collection', frame)

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"\nSuccess! Added {collected_count} samples for '{sign_name}' to {DATA_FILE}.")