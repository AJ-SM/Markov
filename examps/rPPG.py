import cv2 as cv
import mediapipe as mp 



# Path 
path = r"D:\Storeage-1\Main\ML-Model\data_set\NiggaVideo.mp4"
cap = cv.VideoCapture(path)

# 1. Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize the model
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,                # Limit to 1 face for better performance
    refine_landmarks=True,          # Includes iris landmarks (478 total)
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cv.namedWindow("NiggaVideo",cv.WINDOW_NORMAL)
cv.resizeWindow("NiggaVideo",1080,720)



while True:
    ret,frame = cap.read()

    embs = face_mesh.process(frame)
    if not ret:
        break

    cv.imshow("NiggaVideo",frame)
    cv.imshow("NiggaVideo",embs)

    if cv.waitKey(25) & 0xFF ==ord('q'):
        break

cap.release()
cv.destroyAllWindows()