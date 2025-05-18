import os
import cv2
import tempfile
import numpy as np
import streamlit as st
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, VideoFrame
import av

# Constants
CONFIDENCE_THRESHOLD = 0.5
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)
MAX_AGE = 50
BBOX_OFFSET = 20

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

st.title("Real-Time Object Detection & Tracking")

st.sidebar.title("Options")
model_path = st.sidebar.text_input("Enter YOLO model path", "yolov8n.pt")
video_mode = st.sidebar.radio("Select Video Mode", ("Original", "Processed"))

model = YOLO(model_path)

# ---------- IMAGE UPLOAD AND DETECTION ----------
st.subheader("Upload Image for Detection")
image_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if image_file is not None:
    image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), 1)
    detections = model(image)[0]
    for data in detections.boxes.data.tolist():
        confidence = data[4]
        if confidence < CONFIDENCE_THRESHOLD:
            continue
        xmin, ymin, xmax, ymax = map(int, data[:4])
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), GREEN, 2)
        cv2.putText(image, f"{confidence:.2f}", (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 2)
    st.image(image, channels="BGR", caption="Detected Image")

# ---------- VIDEO UPLOAD ----------
st.subheader("Upload Video for Detection & Tracking")
uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
if uploaded_video:
    input_path = os.path.join(UPLOAD_FOLDER, uploaded_video.name)
    with open(input_path, "wb") as f:
        f.write(uploaded_video.read())
    output_path = os.path.join(OUTPUT_FOLDER, f"processed_{uploaded_video.name}")

    if st.button("Track"):
        tracker = DeepSort(max_age=MAX_AGE)
        cap = cv2.VideoCapture(input_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        progress = st.progress(0)

        for idx in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            detections = model(frame)[0]
            results = []
            for data in detections.boxes.data.tolist():
                if data[4] < CONFIDENCE_THRESHOLD:
                    continue
                xmin, ymin, xmax, ymax = map(int, data[:4])
                results.append([[xmin, ymin, xmax - xmin, ymax - ymin], data[4], int(data[5])])
            tracks = tracker.update_tracks(results, frame=frame)
            for track in tracks:
                if not track.is_confirmed():
                    continue
                track_id = track.track_id
                ltrb = track.to_ltrb()
                xmin, ymin, xmax, ymax = map(int, ltrb)
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), GREEN, 2)
                cv2.putText(frame, f"ID: {track_id}", (xmin + 5, ymin - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 2)
            out.write(frame)
            progress.progress((idx + 1) / total_frames)

        cap.release()
        out.release()
        st.success("Processing complete!")
        if video_mode == "Processed":
            st.video(output_path)
        else:
            st.video(input_path)

# ---------- CAMERA VIA BROWSER ----------
st.subheader("Live Camera (Browser Based)")

class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.tracker = DeepSort(max_age=MAX_AGE)

    def recv(self, frame: VideoFrame) -> VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        detections = model(img)[0]
        results = []
        for data in detections.boxes.data.tolist():
            if data[4] < CONFIDENCE_THRESHOLD:
                continue
            xmin, ymin, xmax, ymax = map(int, data[:4])
            results.append([[xmin, ymin, xmax - xmin, ymax - ymin], data[4], int(data[5])])
        tracks = self.tracker.update_tracks(results, frame=img)
        for track in tracks:
            if not track.is_confirmed():
                continue
            ltrb = track.to_ltrb()
            xmin, ymin, xmax, ymax = map(int, ltrb)
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), GREEN, 2)
            cv2.putText(img, f"ID: {track.track_id}", (xmin, ymin - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 2)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

webrtc_streamer(key="object-detect", video_processor_factory=VideoProcessor)
