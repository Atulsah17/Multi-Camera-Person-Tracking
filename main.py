import os
import threading
from queue import Queue
from tracker import process_video

# List of input videos
video_list = [
    "input videos\cam1.mp4",
    "input videos\cam2.mp4",
    "input videos\cam3.mp4",
    "input videos\cam4.mp4",
    "input videos\cam5.mp4"
]

# Queue to track processing status
status_queue = Queue()

def start_processing():
    os.makedirs("output/temp", exist_ok=True)
    threads = []
    for cam_idx, video_path in enumerate(video_list, 1):
        print(f"Starting thread for cam {cam_idx}: {video_path}")
        thread = threading.Thread(target=process_video, args=(cam_idx, video_path, status_queue))
        threads.append(thread)
        thread.start()
    
    return threads

if __name__ == "__main__":
    threads = start_processing()
    for thread in threads:
        thread.join()