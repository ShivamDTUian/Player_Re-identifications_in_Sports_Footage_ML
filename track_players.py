from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2

# Configuration 
MODEL_PATH = "best.pt"
VIDEO_PATH = r"C:\player_reid_project\15sec_input_720p.mp4"
OUTPUT_PATH = r"C:\player_reid_project\tracked_output.mp4"
CONFIDENCE_THRESHOLD = 0.3

# Load Models ---
model = YOLO(MODEL_PATH)
tracker = DeepSort(max_age=20, n_init=3, max_cosine_distance=0.3)

# --- Vid Stp
cap = cv2.VideoCapture(VIDEO_PATH)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count_expected = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

# class names from model
class_names = model.names
PLAYER_LABELS = ['player', 'Player']
BALL_LABELS = ['ball', 'Ball']
REFEREE_LABELS = ['referee', 'Referee']

# For consistent ID remapping
id_map = {}
next_id = 1

# -processing loop
frame_counter = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_counter += 1

    results = model(frame)[0]
    detections = []

    for box in results.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = class_names[class_id]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)
        cv2.putText(frame, f"{label} ({confidence:.2f})", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

        if label.lower() in PLAYER_LABELS and confidence >= CONFIDENCE_THRESHOLD:
            detections.append(([x1, y1, x2, y2], confidence, 'player'))

    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id


        if track_id not in id_map:
            id_map[track_id] = next_id
            next_id += 1
        mapped_id = id_map[track_id]

        l, t, r, b = map(int, track.to_ltrb())
        cv2.rectangle(frame, (l, t), (r, b), (0, 255, 0), 2)
        cv2.putText(frame, f"Player {mapped_id}", (l, t - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    out.write(frame)

cap.release()
out.release()
print(f"✅ Done. Video saved at: {OUTPUT_PATH}")
print(f"Frames processed: {frame_counter} / {frame_count_expected}")
