import os
import streamlit as st
import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Constants
CONFIDENCE_THRESHOLD = 0.5
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)
MAX_AGE = 50
BBOX_OFFSET = 20

# Define folder paths
UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"

# Ensure folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Streamlit app title

st.title("Real-Time Object Detection & Tracking System")



# Sidebar options
st.sidebar.title("Options")
camera_active = st.session_state.get("camera_active", False)
st.session_state["camera_active"] = camera_active
model_path = st.sidebar.text_input("Enter YOLO model path", "yolov8n.pt")
video_mode = st.sidebar.radio("Select Video Mode", ("Original"))


def toggle_camera():
    if not st.session_state["camera_active"]:
        st.session_state["camera_active"] = True
        run_camera()
    else:
        st.session_state["camera_active"] = False


def run_camera():
    stframe = st.empty()
    cap = cv2.VideoCapture(0)
    model = YOLO(model_path)
    tracker = DeepSort(max_age=MAX_AGE)

    while st.session_state["camera_active"] and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO object detection
        detections = model(frame)[0]

        results = []
        # Parse YOLO detections
        for data in detections.boxes.data.tolist():
            confidence = data[4]

            if float(confidence) < CONFIDENCE_THRESHOLD:
                continue

            xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
            class_id = int(data[5])
            results.append([[xmin, ymin, xmax - xmin, ymax - ymin], confidence, class_id])

        # Run DeepSORT tracking
        tracks = tracker.update_tracks(results, frame=frame)
        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            ltrb = track.to_ltrb()

            xmin, ymin, xmax, ymax = int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])
            # Draw bounding box and track ID
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), GREEN, 2)
            cv2.rectangle(frame, (xmin, ymin - BBOX_OFFSET), (xmin + 50, ymin), GREEN, -1)
            cv2.putText(frame, f"ID: {track_id}", (xmin + 5, ymin - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 2)

        # Display the frame
        stframe.image(frame, channels="BGR")

    cap.release()


def upload_video():
    return st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])


def process_video(input_file_path, output_file_path):
    try:
        # Load YOLOv8 model and DeepSORT tracker
        model = YOLO(model_path)
        tracker = DeepSort(max_age=MAX_AGE)

        # Open the uploaded video
        video_cap = cv2.VideoCapture(input_file_path)

        # Get video properties
        fps = int(video_cap.get(cv2.CAP_PROP_FPS))
        frame_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        # Initialize video writer
        writer = cv2.VideoWriter(output_file_path, fourcc, fps, (frame_width, frame_height))

        # Process the video frame by frame
        progress_bar = st.progress(0)
        frame_count = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))

        for frame_idx in range(frame_count):
            ret, frame = video_cap.read()

            if not ret:
                break

            # Run YOLO object detection
            detections = model(frame)[0]

            results = []
            # Parse YOLO detections
            for data in detections.boxes.data.tolist():
                confidence = data[4]

                if float(confidence) < CONFIDENCE_THRESHOLD:
                    continue

                xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
                class_id = int(data[5])
                results.append([[xmin, ymin, xmax - xmin, ymax - ymin], confidence, class_id])

            # Run DeepSORT tracking
            tracks = tracker.update_tracks(results, frame=frame)
            for track in tracks:
                if not track.is_confirmed():
                    continue

                track_id = track.track_id
                ltrb = track.to_ltrb()

                xmin, ymin, xmax, ymax = int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])
                # Draw bounding box and track ID
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), GREEN, 2)
                cv2.rectangle(frame, (xmin, ymin - BBOX_OFFSET), (xmin + 50, ymin), GREEN, -1)
                cv2.putText(frame, f"ID: {track_id}", (xmin + 5, ymin - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 2)

            # Write the processed frame to the output video
            writer.write(frame)

            # Update progress bar
            progress_bar.progress((frame_idx + 1) / frame_count)

        # Release resources
        video_cap.release()
        writer.release()

        return True
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return False


uploaded_video = upload_video()
if uploaded_video:
    # Save the uploaded video to the uploads folder
    input_file_path = os.path.join(UPLOAD_FOLDER, uploaded_video.name)
    if os.path.exists(input_file_path):
        st.warning("File already exists. Overwriting.")
    with open(input_file_path, "wb") as f:
        f.write(uploaded_video.read())

    # Define output file path in the outputs folder
    output_file_path = os.path.join(OUTPUT_FOLDER, f"processed_{uploaded_video.name}")

    if st.button("Track", key="track_button", disabled=False):
        if process_video(input_file_path, output_file_path):
            st.success("Processing completed!")
            if video_mode == "Processed":
                st.video(output_file_path)
            else:
                st.video(input_file_path)

if st.button("Start/Stop Camera"):
    toggle_camera()
