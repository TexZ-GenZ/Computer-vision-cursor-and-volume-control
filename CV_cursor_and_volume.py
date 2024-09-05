import cv2
import mediapipe as mp
import autopy
import time
import numpy as np
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from ctypes import POINTER, cast
from comtypes import CLSCTX_ALL

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.3)
mp_drawing = mp.solutions.drawing_utils

# Initialize webcam
cap = cv2.VideoCapture(1)  # Try 0 first, if it doesn't work, try 1

# Allow some time for the camera to warm up
time.sleep(2)

# Get screen dimensions
screen_width, screen_height = autopy.screen.size()

# Initialize audio interface for volume control
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None
)
volume = cast(interface, POINTER(IAudioEndpointVolume))

# Smoothing parameters for mouse movement
smooth_factor = 0.1
prev_x, prev_y = None, None

# Control frame rate
frame_rate = 30
frame_time = 1000 // frame_rate

# Mouse state
mouse_pressed = False

# Define the maximum distance for 100% volume (in pixels)
max_distance = 200  # Adjust this value based on your needs

# Hand classification state
hand_classified = False
hand_side = None  # 'left' or 'right'

def adjust_volume(distance, max_distance):
    volume_level = np.clip(distance / max_distance, 0.0, 1.0)
    volume.SetMasterVolumeLevelScalar(volume_level, None)
    print(f"Volume set to {volume_level * 100:.0f}%")

# Main loop
while True:
    start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Flip the frame horizontally for a mirror effect
    frame = cv2.flip(frame, 1)

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame and detect hands
    results = hands.process(rgb_frame)

    # Get the height and width of the frame
    height, width, _ = frame.shape

    if results.multi_hand_landmarks:
        print(f"Number of hands detected: {len(results.multi_hand_landmarks)}")
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get the wrist position to classify the hand
            wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
            wrist_x = int(wrist.x * width)

            # Classify hand as left or right based on wrist position
            if wrist_x < width // 2:  # Left hand
                hand_side = 'left'
            else:  # Right hand
                hand_side = 'right'

            # Get thumb and index finger positions
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

            # Convert normalized coordinates to pixel values
            thumb_x, thumb_y = int(thumb_tip.x * width), int(thumb_tip.y * height)
            index_x, index_y = int(index_finger_tip.x * width), int(index_finger_tip.y * height)

            if hand_side == 'left':
                # Mouse control logic
                smoothed_x = int(prev_x + (index_x - prev_x) * smooth_factor) if prev_x is not None else index_x
                smoothed_y = int(prev_y + (index_y - prev_y) * smooth_factor) if prev_y is not None else index_y

                smoothed_x = max(0, min(smoothed_x, width - 1))
                smoothed_y = max(0, min(smoothed_y, height - 1))

                autopy.mouse.move(screen_width * (smoothed_x / width), screen_height * (smoothed_y / height))

                if abs(thumb_x - index_x) < 30 and abs(thumb_y - index_y) < 30:
                    if not mouse_pressed:
                        autopy.mouse.click()
                        mouse_pressed = True
                        print("Mouse clicked")
                else:
                    mouse_pressed = False

                prev_x, prev_y = smoothed_x, smoothed_y

                cv2.circle(frame, (thumb_x, thumb_y), 10, (0, 0, 255), -1)  # Thumb
                cv2.circle(frame, (index_x, index_y), 10, (0, 255, 0), -1)  # Index finger

            elif hand_side == 'right':
                # Volume control logic
                distance = np.sqrt((thumb_x - index_x) ** 2 + (thumb_y - index_y) ** 2)
                adjust_volume(distance, max_distance)

                cv2.circle(frame, (thumb_x, thumb_y), 10, (255, 0, 0), -1)  # Thumb
                cv2.circle(frame, (index_x, index_y), 10, (0, 255, 255), -1)  # Index finger

            # Display hand side on frame
            cv2.putText(frame, f"Hand: {hand_side}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    else:
        print("No hands detected")
        hand_side = None
        prev_x, prev_y = None, None

    # Show the frame
    cv2.imshow("Mouse and Volume Control", frame)

    # Control frame rate
    elapsed_time = time.time() - start_time
    if elapsed_time < frame_time / 1000:
        time.sleep(frame_time / 1000 - elapsed_time)

    # Exit on pressing 'Esc'
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release resources
cap.release()
cv2.destroyAllWindows()