# Multi-Camera Person Tracking

This project implements a multi-camera person tracking system that identifies and tracks individuals across multiple video feeds. It uses YOLOv8 for person detection, ByteTrack for tracking within each camera, and a ResNet50-based ReID model for matching persons across cameras. The system processes videos in chunks, annotates tracked persons with global IDs, and saves the output as video files.

## Features
- **Person Detection**: Uses YOLOv8 to detect persons in each video frame.
- **Tracking**: Employs ByteTrack for robust tracking within each camera feed.
- **Re-Identification (ReID)**: Matches persons across cameras using a ResNet50-based ReID model.
- **Multi-Threaded Processing**: Processes multiple camera feeds in parallel.
- **GPU Support**: Leverages CUDA for faster processing if a compatible GPU is available.
- **Output**: Saves processed videos with tracking annotations as chunks and final combined videos.

## Requirements
- **Operating System**: Windows, macOS, or Linux
- **Hardware**: A CUDA-compatible GPU is recommended for faster processing (CPU fallback available).
- **Python Version**: Python 3.8 or higher
- **Dependencies**:
  - `torch` (with CUDA support if using GPU)
  - `torchvision`
  - `ultralytics`
  - `opencv-python-headless`
  - `numpy`
  - `scipy`
  - ByteTrack (installed via GitHub repository)

## Installation
1. **Clone the Repository** (if applicable):
   ```bash
   git clone <repository-url>
   cd multi-camera-person-tracking
   ```
   Alternatively, place the script (`app.py`) in your project directory.

2. **Set Up a Virtual Environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  
   ```

3. **Install Dependencies**:
   ```bash
   pip install torch torchvision ultralytics opencv-python-headless numpy scipy
   ```
   - If you have a CUDA-compatible GPU, ensure `torch` is installed with CUDA support:
     ```bash
     pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
     ```
     Check your CUDA version and adjust the URL accordingly (e.g., `cu118` for CUDA 11.8).

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
- `input_videos/`: Directory to place your input video files.
- `output/`: Directory where processed videos and temporary chunks are saved.
  - `output/temp/camX/`: Temporary chunks for each camera (e.g., `chunk_0.mp4`).
  - `output/output_camX.mp4`: Final processed video for each camera.

## Usage
1. **Prepare Input Videos**:
   - Place your input videos in the `input_videos/` directory.
   - Name the videos as `cam1.mp4`, `cam2.mp4`, ..., `cam5.mp4` (for 5 cameras).
   - Example:
     ```
     input_videos/
     ├── cam1.mp4
     ├── cam2.mp4
     ├── cam3.mp4
     ├── cam4.mp4
     ├── cam5.mp4
     ```

2. **Update Video Paths** (if needed):
   - Open `multi_camera_tracking_local.py` and modify the `video_list` to match your video paths:
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
   ```bash
   python multi_camera_tracking_local.py
   ```
   - The script will:
     - Detect and track persons in each video.
     - Assign global IDs to match persons across cameras.
     - Save video chunks every 60 frames to `output/temp/camX/`.
     - Combine chunks into final videos at `output/output_camX.mp4`.

4. **Monitor Progress**:
   - The script prints progress updates, such as frame counts and FPS for each camera.
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

## Contact
For questions or contributions, please reach out to [atulsah9211@gmail.com].