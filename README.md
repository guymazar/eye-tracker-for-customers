# Eye Tracking Project

## Overview
This project, "Understanding Customer Attention through Eye Tracking," is designed to analyze user attention using a single RGB camera. It detects and tracks eye movements to generate gaze heatmaps, helping understand customer behavior in various scenarios.

## Features
- Real-time eye tracking using OpenCV
- Face and eye detection with Haar cascades
- Gaze estimation and visualization
- Heatmap generation for gaze analysis
- Modular and extensible codebase

## Folder Structure
```
EyeTrackingProject/
│── src/                    # Source code
│   │── main.py             # Entry point for running the eye-tracking system
│   │── eye_tracker.py      # Eye tracking implementation
│   │── face_detection.py   # Face and eye detection utilities
│   │── utils.py            # Helper functions
│
│── models/                 # Pre-trained models (Haar cascades, etc.)
│── data/                   # Sample images/videos for testing
│── output/                 # Results and logs (heatmaps, logs, etc.)
│── tests/                  # Unit tests
│── docs/                   # Documentation
│── requirements.txt        # Dependencies
│── setup.py                # Optional: If packaging is needed
```

## Installation
### Prerequisites
Ensure you have Python 3.8+ installed. Install dependencies using:
```bash
pip install -r requirements.txt
```

## Running the Project
1. Run the main script to start eye tracking:
```bash
python src/main.py
```
2. To analyze a specific video, use:
```bash
python src/main.py --video data/test_video.mp4
```

## License
This project is for academic and research purposes.

---

# Instructions for Running the Eye Tracking Project

## Step 1: Clone the Repository
If this project is hosted on GitHub, clone it using:
```bash
git clone https://github.com/your-repo/EyeTrackingProject.git
cd EyeTrackingProject
```

## Step 2: Install Dependencies
Run:
```bash
pip install -r requirements.txt
```

## Step 3: Run the Eye Tracker
To start real-time eye tracking using the webcam:
```bash
python src/main.py
```

To analyze a video file instead:
```bash
python src/main.py --video data/test_video.mp4
```

## Step 4: Viewing Outputs
- Gaze heatmaps are stored in the `output/gaze_heatmaps/` folder.
- Logs are available in the `output/logs/` directory.

## Step 5: Running Tests
To validate the implementation, run:
```bash
pytest tests/
```

## Troubleshooting
If OpenCV-related errors occur, try:
```bash
pip install opencv-python opencv-contrib-python
```

