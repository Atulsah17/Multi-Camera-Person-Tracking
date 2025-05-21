from flask import Flask, render_template, jsonify
import os
import time
from main import status_queue, video_list, start_processing

app = Flask(__name__)

# Track processing status
camera_status = {i: {"status": "pending", "progress": "", "chunks": []} for i in range(1, 6)}

def update_status():
    while True:
        try:
            cam_idx, status_type, message = status_queue.get_nowait()
            print(f"Queue update at {time.strftime('%H:%M:%S')}: cam {cam_idx}, type {status_type}, message {message}")
            if status_type == "chunk":
                rel_path = os.path.relpath(message, start='static').replace('\\', '/')
                camera_status[cam_idx]["chunks"].append(rel_path)
                print(f"Added chunk for cam {cam_idx}: {rel_path}, full URL: http://localhost:5000/static/{rel_path}")
            elif status_type == "progress":
                camera_status[cam_idx]["progress"] = message
            elif status_type == "completed":
                camera_status[cam_idx]["status"] = "completed"
                rel_path = os.path.relpath(message, start='static').replace('\\', '/')
                camera_status[cam_idx]["final_video"] = rel_path
                print(f"Final video for cam {cam_idx}: {rel_path}, full URL: http://localhost:5000/static/{rel_path}")
            elif status_type == "error":
                camera_status[cam_idx]["status"] = "error"
                camera_status[cam_idx]["progress"] = message
        except:
            break

@app.route('/')
def index():
    update_status()
    return render_template('index.html', cameras=camera_status)

@app.route('/status')
def status():
    update_status()
    print(f"Status at {time.strftime('%H:%M:%S')}: {camera_status}")
    return jsonify(camera_status)

if __name__ == '__main__':
    os.makedirs("static/temp/cam1", exist_ok=True)
    os.makedirs("static/temp/cam2", exist_ok=True)
    os.makedirs("static/temp/cam3", exist_ok=True)
    os.makedirs("static/temp/cam4", exist_ok=True)
    os.makedirs("static/temp/cam5", exist_ok=True)
    print(f"Starting video processing at {time.strftime('%H:%M:%S')}")
    start_processing()
    app.run(debug=True, host='0.0.0.0', port=5000)