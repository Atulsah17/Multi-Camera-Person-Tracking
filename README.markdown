# Multi-Camera Person Tracking

This Multi-Camera Person Tracking project uses YOLOv8 for detection, ByteTrack for tracking, and ResNet50-based ReID to match individuals across video feeds. It processes videos in parallel, annotates persons with global IDs, and saves outputs as chunks and final videos, ideal for surveillance and monitoring applications.

## Features
- **Person Detection**: Uses YOLOv8 to detect persons in each video frame.
- **Tracking**: Employs ByteTrack for robust tracking within each camera feed.
- **Re-Identification (ReID)**: Matches persons across cameras using a ResNet50-based ReID model.
- **Multi-Threaded Processing**: Processes multiple camera feeds in parallel.
- **Output**: Saves processed videos with tracking annotations as chunks and final combined videos.

## Requirements
- **Operating System**: Windows, macOS, or Linux
- **Hardware**: A CUDA-compatible GPU is recommended for faster processing (CPU fallback available).
- **Python Version**: Python 3.8 or higher
- **Dependencies**:
  - See `requirements.txt` for most dependencies.
  - ByteTrack (installed manually via GitHub repository).

## Installation
1. **Clone the Repository** (if applicable):
   ```bash
   git clone <https://github.com/Atulsah17/Multi-Camera-Person-Tracking>
   cd multi-camera-person-tracking
   ```
   Alternatively, place the script (`multi_camera_tracking_local.py`) in your project directory.

2. **Set Up a Virtual Environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies from `requirements.txt`**:
   ```bash
   pip install -r requirements.txt
   ```
   - If you have a CUDA-compatible GPU, ensure `torch` is installed with CUDA support. The `requirements.txt` specifies `torch==2.0.1`, which may need a specific CUDA version (e.g., CUDA 11.7). Check your CUDA version and adjust if necessary:
     ```bash
     pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu117
     ```

4. **Install ByteTrack**:
   ByteTrack requires manual installation from its GitHub repository:
   ```bash
   git clone https://github.com/ifzhang/ByteTrack.git
   cd ByteTrack
   python setup.py install
   cd ..
   ```
   - Ensure you have `setuptools` and other build tools installed:
     ```bash
     pip install setuptools
     ```
   - If you encounter issues, ensure `torch` is installed before running `setup.py`.

5. **Verify GPU Support** (optional):
   Run the following to confirm PyTorch detects your GPU:
   ```python
   import torch
   print(torch.cuda.is_available())
   ```
   Should output `True` if GPU support is set up correctly.

## Project Structure
- `multi_camera_tracking_local.py`: Main script for processing videos and tracking persons.
- `requirements.txt`: List of pip-installable dependencies.
- `input_videos/`: Directory to place your input video files.
- `output/`: Directory where processed videos and temporary chunks are saved.
  - `output/temp/camX/`: Temporary chunks for each camera (e.g., `chunk_0.mp4`).
  - `output/output_camX.mp4`: Final processed video for each camera.

## How to Run the Code
1. **Prepare Input Videos**:
   - Place your input videos in the `input_videos/` directory.
   - Name the videos as `cam1.mp4`, `cam2.mp4`, ..., `cam5.mp4` (for 5 cameras).
   - Example directory structure:
     ```
     input_videos/
     ├── cam1.mp4
     ├── cam2.mp4
     ├── cam3.mp4
     ├── cam4.mp4
     ├── cam5.mp4
     ```

2. **Update Video Paths** (if needed):
   - Open `multi_camera_tracking_local.py` in a text editor.
   - Modify the `video_list` variable to match the paths to your input videos:
     ```python
     video_list = [
         "input_videos/cam1.mp4",
         "input_videos/cam2.mp4",
         "input_videos/cam3.mp4",
         "input_videos/cam4.mp4",
         "input_videos/cam5.mp4"
     ]
     ```

3. **Run the Script**:
   - Ensure your virtual environment is activated:
     ```bash
     source venv/bin/activate  # On Windows: venv\Scripts\activate
     ```
   - Execute the script:
     ```bash
     python multi_camera_tracking_local.py
     ```
   - The script will process each video in parallel, printing progress updates like frame counts and FPS for each camera.
   - Example output:
     ```
     Using device: cuda
     Starting thread for cam 1: input_videos/cam1.mp4
     Starting thread for cam 2: input_videos/cam2.mp4
     ...
     Cam 1, Frame 30, FPS: 29.50
     Creating chunk: output/temp/cam1/chunk_0.mp4 at 19:45:00
     Processed input_videos/cam1.mp4, Avg FPS: 30.00
     ```

## Method Explanation
### Detection
- **Model**: Uses YOLOv8 (medium variant, `yolov8m.pt`) for person detection.
- **Process**: Each video frame is processed to detect persons (class ID 0) with a confidence threshold of 0.7.
- **Output**: Bounding boxes and confidence scores for detected persons in each frame.

### Tracking
- **Model**: Employs ByteTrack for tracking persons within each camera feed.
- **Process**: 
  - Takes YOLOv8 detections as input.
  - Tracks persons across frames using a combination of Kalman filtering and appearance features.
  - Assigns local track IDs to persons within each video.
- **Configuration**: Uses a tracking threshold of 0.4, match threshold of 0.7, and a buffer of 60 frames to handle occlusions.

### Re-Identification (ReID)
- **Model**: Uses a ResNet50-based ReID model (pre-trained on ImageNet) for appearance feature extraction.
- **Process**:
  - Crops detected persons from frames and resizes them to 256x128 pixels.
  - Extracts feature embeddings using ResNet50 with an adaptive average pooling layer.
  - Normalizes features and computes cosine similarity to match persons across cameras.
  - Assigns global IDs to persons by comparing features against a database of known identities.
- **Thresholds**: Uses a similarity threshold of 0.4 for cross-camera matching and 0.85 to avoid over-matching within the same video.

## Output
- **Temporary Chunks**: Saved in `output/temp/camX/chunk_Y.mp4` (e.g., `output/temp/cam1/chunk_0.mp4`).
  - Each chunk contains 60 frames of processed video with tracking annotations.
- **Final Videos**: Saved in `output/output_camX.mp4` (e.g., `output/output_cam1.mp4`).
  - These are the combined videos for each camera, with green bounding boxes and global IDs (e.g., `GID 1`) drawn on tracked persons.

## Troubleshooting
- **Input Video Not Found**:
  - Ensure videos are in the `input_videos/` directory and paths in `video_list` are correct.
- **Slow Processing**:
  - Confirm GPU usage (`Using device: cuda`).
  - Use a lighter YOLO model (e.g., `yolov8n.pt` instead of `yolov8m.pt`):
    ```python
    model = YOLO('yolov8n.pt').to(device)
    ```
- **Video Playback Issues**:
  - If output videos don’t play, they may need re-encoding for compatibility:
    ```bash
    pip install ffmpeg-python
    ```
    Then re-encode using FFmpeg (example for cam1):
    ```python
    import ffmpeg
    stream = ffmpeg.input('output/output_cam1.mp4')
    stream = ffmpeg.output(stream, 'output/output_cam1_reencoded.mp4', vcodec='h264', acodec='aac', format='mp4')
    ffmpeg.run(stream)
    ```
- **ByteTrack Installation Errors**:
  - Ensure `torch` and `torchvision` are installed before running `setup.py`.
  - If errors persist, check for missing dependencies like `cython`:
    ```bash
    pip install cython
    ```
    Then rerun the ByteTrack installation:
    ```bash
    cd ByteTrack
    python setup.py install
    cd ..
    ```

## Limitations
- **Video Format**: Input videos should be in a compatible format (e.g., MP4). Re-encode if necessary.
- **Memory Usage**: Processing multiple videos in parallel may require significant GPU memory.
- **ReID Accuracy**: Matching accuracy across cameras depends on lighting, angles, and person appearance.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details (if applicable).

## Contact
<<<<<<< HEAD
For questions or contributions, please reach out to [your-email@example.com].
=======
For questions or contributions, please reach out to [atulsah9211@gmail.com].
>>>>>>> 84a951745404ee788f9126bc44bb24f2e756ddbb
