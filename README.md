# Eye Tracking Project

## Overview
This project, "Understanding Customer Attention through Eye Tracking," is designed to analyze user attention using a single RGB camera. It detects and tracks eye movements to generate gaze heatmaps, helping understand customer behavior in various scenarios.

## Features
- Real-time eye tracking using OpenCV
- Face and eye detection with Haar cascades
- MediaPipe Face Mesh support for precise eye landmark detection
- Head pose estimation to correct gaze drift
- Advanced gaze heatmap visualization:
  - Temporal decay for recent attention focus
  - Cumulative heatmaps for overall patterns
  - Gaussian blur for natural, intuitive visualization
- Interactive controls for heatmap management
- Multiple screen layouts for UI/UX testing:
  - Default product showcase
  - Grid layout with labeled zones
  - Product comparison matrix
  - Checkout page
- Region-based analytics with attention metrics
- Modular and extensible codebase

## Folder Structure
```
EyeTrackingProject/
│── src/                    # Source code
│   │── main.py             # Entry point for running the eye-tracking system
│   │── eye_tracker.py      # Eye tracking implementation
│   │── face_detection.py   # Face and eye detection utilities
│   │── face_mesh_tracker.py # MediaPipe face mesh implementation
│   │── gaze_analysis.py    # Gaze analysis and heatmap generation
│   │── screen_layout.py    # Screen layout manager for simulated interfaces
│   │── utils.py            # Helper functions
│
│── models/                 # Pre-trained models (Haar cascades, etc.)
│── data/                   # Sample images/videos for testing
│── output/                 # Results and logs (heatmaps, logs, etc.)
│   │── gaze_heatmaps/      # Saved heatmap images
│   │── logs/               # Log files
│   │── screen_analytics/   # Region-based analytics data and visualizations
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
1. Run the main script with the default settings:
```bash
python src/main.py
```

2. To analyze a specific video:
```bash
python src/main.py --video data/test_video.mp4
```

3. To use traditional eye tracking (without MediaPipe):
```bash
python src/main.py --traditional
```

4. To start with the temporal heatmap display:
```bash
python src/main.py --temporal
```

5. To specify a starting layout:
```bash
python src/main.py --layout grid  # Options: default, grid, product_comparison, checkout
```

6. To disable automatic heatmap reset (default: 30 seconds):
```bash
python src/main.py --auto-reset 0
```

7. Combine multiple options:
```bash
python src/main.py --video data/test_video.mp4 --temporal --layout product_comparison --auto-reset 60
```

## Keyboard Controls
During application execution, you can use the following keyboard controls:
- **Q**: Quit the application
- **R**: Reset the heatmap manually
- **T**: Toggle between temporal and cumulative heatmap displays
- **H**: Toggle help text display
- **L**: Switch to the next screen layout
- **A**: Toggle analytics overlay (showing region-based attention metrics)
- **S**: Save current analytics to CSV and image files

## Understanding the Heatmaps
- **Temporal Heatmap**: Shows recent attention patterns with older points fading over time. This is useful for tracking the current focus of attention.
- **Cumulative Heatmap**: Shows the sum of all gaze points since the last reset. This is useful for identifying overall patterns over time.

## Screen Layouts
The application includes several screen layouts for testing different UI designs:

1. **Default Layout**: Basic product showcase with three products and action buttons
2. **Grid Layout**: A 3x3 grid with labeled zones (A through I) for basic attention mapping
3. **Product Comparison**: A matrix layout comparing three products across different attributes (price, features, etc.)
4. **Checkout Page**: A simulated checkout page with cart summary, customer information, and payment options

Each layout supports region-based analytics, tracking which UI elements receive the most attention.

## Analytics
The system tracks attention metrics for each defined region in the current layout:
- **Gaze Points**: The number of times a user looked at a region
- **Total Time**: The accumulated time spent looking at a region
- **Heatmap Overlay**: Visualizes attention intensity across different UI elements

Analytics can be saved to CSV files for later analysis, along with heatmap images.

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

## Step 2: Create a Virtual Environment (Recommended)
```bash
python3 -m venv eyetracker_venv
source eyetracker_venv/bin/activate  # On Windows: eyetracker_venv\Scripts\activate
```

## Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

## Step 4: Run the Eye Tracker
To start real-time eye tracking using the webcam:
```bash
python src/main.py
```

To analyze a video file instead:
```bash
python src/main.py --video data/test_video.mp4
```

## Step 5: Viewing Outputs
- Both temporal and cumulative heatmaps are stored in the `output/gaze_heatmaps/` folder.
- Logs are available in the `output/logs/` directory.
- Region-based analytics are saved to `output/screen_analytics/`.

## Step 6: Running Tests
To validate the implementation, run:
```bash
pytest tests/
```

## Troubleshooting
If OpenCV-related errors occur, try:
```bash
pip install opencv-python opencv-contrib-python
```

If webcam permission issues occur:
1. Check your system's camera privacy settings
2. Ensure your application has permission to access the camera

For MediaPipe installation issues:
```bash
pip install --upgrade mediapipe
```

