import cv2
import torch
import numpy as np
import time
import os
import shutil
from ultralytics import YOLO
from ByteTrack.yolox.tracker.byte_tracker import BYTETracker
from types import SimpleNamespace
from reid import ReIDWrapper, reid_transform, extract_features, match_person, track_features_buffer
from utils import compute_iou

# Device setup
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load YOLOv8 model
model = YOLO('yolov8m.pt').to(device)

# Tracker arguments
def create_args(frame_rate=30):
    args = SimpleNamespace()
    args.track_thresh = 0.4
    args.track_buffer = 60
    args.match_thresh = 0.7
    args.min_box_area = 100
    args.mot20 = False
    return args

def process_video(cam_idx, video_path, status_queue):
    if not os.path.exists(video_path):
        print(f"Input video not found: {video_path}")
        status_queue.put((cam_idx, "error", "Input video not found"))
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open {video_path}")
        status_queue.put((cam_idx, "error", "Failed to open video"))
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    output_path = f"output/output_cam{cam_idx}.mp4"
    temp_dir = f"output/temp/cam{cam_idx}"
    static_temp_dir = f"static/temp/cam{cam_idx}"
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(static_temp_dir, exist_ok=True)

    tracker = BYTETracker(create_args(fps), frame_rate=fps)
    frame_idx = 0
    start_time = time.time()
    temp_writer = None
    temp_frame_count = 0
    chunk_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Initialize temporary writer for new chunk
        if temp_frame_count == 0:
            if temp_writer is not None:
                temp_writer.release()
                print(f"Released writer for chunk {chunk_idx - 1} at {time.strftime('%H:%M:%S')}")
            chunk_path = f"{temp_dir}/chunk_{chunk_idx}.mp4"
            static_chunk_path = f"{static_temp_dir}/chunk_{chunk_idx}.mp4"
            print(f"Creating chunk: {chunk_path} at {time.strftime('%H:%M:%S')}")
            temp_writer = cv2.VideoWriter(chunk_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
            if not temp_writer.isOpened():
                print(f"Failed to open VideoWriter for {chunk_path}")
                status_queue.put((cam_idx, "error", "Failed to create video chunk"))
                return

        # Detect people
        results = model(frame, classes=[0], device=device, conf=0.7)
        detections = [[int(box.xyxy[0][0]), int(box.xyxy[0][1]), int(box.xyxy[0][2]), int(box.xyxy[0][3]), box.conf[0].item()] for result in results for box in result.boxes]

        if not detections:
            temp_writer.write(frame)
            frame_idx += 1
            temp_frame_count += 1
            if temp_frame_count >= 30:
                temp_writer.release()
                print(f"Released writer for chunk {chunk_idx} at {time.strftime('%H:%M:%S')}")
                time.sleep(2)  # Increased delay to ensure file is released
                for attempt in range(5):  # Increased retries
                    try:
                        shutil.copy2(chunk_path, static_chunk_path)  # Use copy2 to preserve metadata
                        print(f"Copied chunk to: {static_chunk_path} at {time.strftime('%H:%M:%S')}")
                        break
                    except Exception as e:
                        print(f"Attempt {attempt + 1} - Failed to copy chunk {chunk_path} to {static_chunk_path}: {e}")
                        time.sleep(2)
                else:
                    print(f"Failed to copy chunk after retries: {chunk_path}")
                    status_queue.put((cam_idx, "error", "Failed to copy video chunk"))
                    return
                status_queue.put((cam_idx, "chunk", static_chunk_path))
                temp_frame_count = 0
                chunk_idx += 1
            continue

        # Track within video
        dets = torch.tensor(detections)
        online_targets = tracker.update(dets, [height, width], [height, width])
        existing_track_ids = [track.track_id for track in online_targets]
        tracks_info = []

        for track in online_targets:
            x1, y1, w, h = map(int, track.tlwh)
            track_bbox = [x1, y1, x1 + w, y1 + h]
            best_iou, best_det_idx = 0, 0
            for i, det in enumerate(detections):
                iou = compute_iou(track_bbox, det[:4])
                if iou > best_iou:
                    best_iou, best_det_idx = iou, i
            tracks_info.append({'track_id': track.track_id, 'bbox': track_bbox, 'confidence': detections[best_det_idx][4]})

        # Extract features and assign global IDs
        bboxes = [info['bbox'] for info in tracks_info]
        confidences = [info['confidence'] for info in tracks_info]
        track_ids = [info['track_id'] for info in tracks_info]
        features_list = extract_features(frame, bboxes, confidences)

        used_gids_this_frame = set()
        for track_id, feat in zip(track_ids, features_list):
            if feat is None:
                continue
            track_features_buffer[(cam_idx, track_id)].append(feat)
            agg_feat = np.mean(np.array(track_features_buffer[(cam_idx, track_id)]), axis=0)
            global_id = match_person(agg_feat, cam_idx, track_id, existing_track_ids, used_gids_this_frame)
            if global_id == -1 or global_id in used_gids_this_frame:
                continue
            used_gids_this_frame.add(global_id)
            x1, y1, x2, y2 = map(int, bboxes[track_ids.index(track_id)])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'GID {global_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        temp_writer.write(frame)
        frame_idx += 1
        temp_frame_count += 1
        if temp_frame_count >= 30:
            temp_writer.release()
            print(f"Released writer for chunk {chunk_idx} at {time.strftime('%H:%M:%S')}")
            time.sleep(2)  # Increased delay
            for attempt in range(5):
                try:
                    shutil.copy2(chunk_path, static_chunk_path)
                    print(f"Copied chunk to: {static_chunk_path} at {time.strftime('%H:%M:%S')}")
                    break
                except Exception as e:
                    print(f"Attempt {attempt + 1} - Failed to copy chunk {chunk_path} to {static_chunk_path}: {e}")
                    time.sleep(2)
            else:
                print(f"Failed to copy chunk after retries: {chunk_path}")
                status_queue.put((cam_idx, "error", "Failed to copy video chunk"))
                return
            status_queue.put((cam_idx, "chunk", static_chunk_path))
            temp_frame_count = 0
            chunk_idx += 1

        if frame_idx % 30 == 0:
            elapsed = time.time() - start_time
            status_queue.put((cam_idx, "progress", f"Frame {frame_idx}, FPS: {frame_idx / elapsed:.2f}"))

    if temp_writer is not None:
        temp_writer.release()
        print(f"Released final writer for cam {cam_idx} at {time.strftime('%H:%M:%S')}")
    cap.release()

    # Combine chunks into final video
    final_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    for i in range(chunk_idx + 1):
        chunk_path = f"{temp_dir}/chunk_{i}.mp4"
        if os.path.exists(chunk_path):
            cap = cv2.VideoCapture(chunk_path)
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                final_writer.write(frame)
            cap.release()
    final_writer.release()

    static_final_path = f"static/output_cam{cam_idx}.mp4"
    for attempt in range(5):
        try:
            shutil.copy2(output_path, static_final_path)
            print(f"Copied final video to: {static_final_path} at {time.strftime('%H:%M:%S')}")
            break
        except Exception as e:
            print(f"Attempt {attempt + 1} - Failed to copy final video {output_path} to {static_final_path}: {e}")
            time.sleep(2)
    else:
        print(f"Failed to copy final video after retries: {output_path}")
        status_queue.put((cam_idx, "error", "Failed to copy final video"))
        return
    status_queue.put((cam_idx, "completed", static_final_path))
    print(f"Processed {video_path}, Avg FPS: {frame_idx / (time.time() - start_time):.2f}")