import cv2 as cv
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Path 
path = r"D:\Storeage-1\Main\ML-Model\data_set\NiggaVideo.mp4"
cap = cv.VideoCapture(path)

# Load model
base_options = python.BaseOptions(model_asset_path="./face_landmarker.task")

options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=True,
    output_facial_transformation_matrixes=True,
    num_faces=2
)

detector = vision.FaceLandmarker.create_from_options(options)


cv.namedWindow("NiggaVideo", cv.WINDOW_NORMAL)
cv.resizeWindow("NiggaVideo", 1080, 720)

while True:
    ret, frame = cap.read()
    if not ret: 
        break
    
    # Convert the BGR frame to RGB for MediaPipe
    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    
    # Process the frame
    rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = detector.detect(mp_image)



    if result.face_landmarks:
        for face_landmarks in result.face_landmarks:
            for landmark in face_landmarks:
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                cv.circle(frame, (x, y), 1, (0,255,0), -1)

    cv.imshow("FaceMesh", frame)

    if cv.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()