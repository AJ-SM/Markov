from pyVHR.extraction.sig_processing import SignalProcessing
from pyVHR.BVP.methods import cpu_POS

video_file = "face_video.mp4"

sig = SignalProcessing()

# Extract RGB signals from video
rgb = sig.extract_holistic(video_file)

# Apply POS rPPG algorithm
bvp = cpu_POS(rgb)

print("BVP length:", len(bvp))