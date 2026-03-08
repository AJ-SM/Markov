import cv2 as cv
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from scipy.signal import detrend, butter, filtfilt
from scipy.fft import fft, fftfreq
from collections import deque

# --- CONFIGURATION ---
# VIDEO_PATH = r"D:\Storeage-1\Main\ML-Model\data_set\pandu.mp4"
# VIDEO_PATH = r"D:\Storeage-1\Main\ML-Model\data_set\NiggaVideo.mp4"
VIDEO_PATH = r"D:\Storeage-1\Main\ML-Model\data_set\udy.mp4"
# VIDEO_PATH = r"D:\Storeage-1\Main\ML-Model\data_set\udyF.mp4"
# VIDEO_PATH = r"D:\Storeage-1\Main\ML-Model\data_set\lady.mp4"
MODEL_PATH = "./face_landmarker.task"
BUFFER_SIZE = 200  # Roughly 5 seconds of data at 30fps

# --- INITIALIZATION ---
cap = cv.VideoCapture(VIDEO_PATH)
fps = int(cap.get(cv.CAP_PROP_FPS)) or 10
roi_buffer = deque(maxlen=BUFFER_SIZE)

# MediaPipe Setup
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1
)
detector = vision.FaceLandmarker.create_from_options(options)

def calculate_rppg(buffer):
    """Processes the buffered ROI data to calculate E and Gamma."""
    data = np.array(buffer)
    # Extract channels [B, G, R]
    b, g, r = data[:, 0], data[:, 1], data[:, 2]
    
    # Detrend to remove DC component (skin tone variations)
    r, g, b = detrend(r), detrend(g), detrend(b)
    
    # Bandpass Filter (0.7Hz - 4Hz = 42bpm - 240bpm)
    nyq = 0.5 * fps
    low, high = 0.7 / nyq, 4 / nyq
    b_filt, a_filt = butter(3, [low, high], btype='band')
    
    g_filt = filtfilt(b_filt, a_filt, g)
    
    # FFT
    freqs = fftfreq(len(g_filt), d=1/fps)
    fft_values = np.abs(fft(g_filt))
    
    # Calculate E (Energy) and Gamma (Spectral concentration)
    band = (freqs >= 0.7) & (freqs <= 4)
    if np.sum(band) > 0:
        band_values = fft_values[band]
        E = np.max(band_values)
        Gamma = E / (np.sum(band_values) + 1e-6) 
        return E, Gamma
    return 0, 0

# --- MAIN LOOP ---
timestamp_ms = 0
cv.namedWindow("rPPG Tracker", cv.WINDOW_NORMAL)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    height, width, _ = frame.shape
    mask = np.zeros((height, width), dtype=np.uint8)
    
    rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    
    result = detector.detect_for_video(mp_image, timestamp_ms)
    timestamp_ms += int(1000 / fps)

    E, Gamma = 0, 0
    output_frame = np.zeros_like(frame)

    if result.face_landmarks:
        for face_landmarks_list in result.face_landmarks:
            points = []
            for landmark in face_landmarks_list:
                if landmark.y > 0.4: 
                    points.append((int(landmark.x * width), int(landmark.y * height)))
            
            if points:
                x_coords = [p[0] for p in points]
                y_coords = [p[1] for p in points]
                cv.rectangle(mask, (min(x_coords), min(y_coords)), 
                             (max(x_coords), max(y_coords)), 255, -1)
                
                # Apply mask to frame
                output_frame = cv.bitwise_and(frame, frame, mask=mask)
                
                # Get mean ROI color
                mean_bgr = cv.mean(frame, mask=mask)[:3]
                roi_buffer.append(mean_bgr)
                
                # Calculate metrics if buffer is full
                if len(roi_buffer) == BUFFER_SIZE:
                    E, Gamma = calculate_rppg(roi_buffer)

    # Display Overlay
    text = f"E: {E:.2f}  Gamma: {Gamma:.4f}"
    cv.putText(output_frame, text, (20, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    cv.imshow("rPPG Tracker", output_frame)
    if cv.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv.destroyAllWindows()