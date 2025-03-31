# Eye Tracking Project

## Overview
This project, "Understanding Customer Attention through Eye Tracking," is designed to analyze user attention using a single RGB camera. It detects and tracks eye movements to generate gaze heatmaps, helping understand customer behavior in various scenarios.

## Features
- Real-time eye tracking using OpenCV
- Face and eye detection with Haar cascades
- MediaPipe Face Mesh support for precise eye landmark detection
- **New Pupil-Based Calibration System** for highly accurate gaze tracking
- Head pose estimation to correct gaze drift
- Smoothed gaze tracking with reduced sensitivity for stability
- Full HD display (1920x1080) that centers on your screen
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
│── output/                 # Results and logs (heatmaps, logs, etc.)
│   │── gaze_heatmaps/      # Saved heatmap images
│   │── logs/               # Log files
│   │── screen_analytics/   # Region-based analytics data and visualizations
│── tests/                  # Unit tests
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
For best results, we recommend using the new calibration system:

```bash
python src/main.py 
```

### Common Options:
1. Run with default settings and calibration:
```bash
python src/main.py
```

2. For Mac users with camera warnings:
```bash
python src/main.py --mac-fix
```

3. To analyze a specific video:
```bash
python src/main.py --video data/test_video.mp4 --skip-calibration
```

4. To use traditional eye tracking (without MediaPipe):
```bash
python src/main.py --traditional
```

5. To start with the temporal heatmap display:
```bash
python src/main.py --temporal
```

6. To specify a starting layout:
```bash
python src/main.py --layout grid  # Options: default, grid, product_comparison, checkout
```

7. To disable automatic heatmap reset (default: 30 seconds):
```bash
python src/main.py --auto-reset 0
```

8. For debugging issues:
```bash
python src/main.py --debug
```

9. To skip calibration (not recommended for best accuracy):
```bash
python src/main.py --skip-calibration
```

10. Combine multiple options:
```bash
python src/main.py --video data/test_video.mp4 --temporal --layout product_comparison --mac-fix
```

## New Pupil-Based Calibration System

The system now features a comprehensive pupil calibration process that significantly improves tracking accuracy:

### How Calibration Works
1. **5-Point Calibration**: The system guides you through looking at 5 points on the screen:
   - Center
   - Top Left
   - Top Right
   - Bottom Left
   - Bottom Right

2. **Pupil Position Mapping**: For each point, the system records the exact position of your pupils in both eyes.

3. **Range Calculation**: After calibration, the system knows your full range of eye movement and can map pupil positions to screen coordinates with high precision.

4. **Personal Adaptation**: This calibration adapts to your unique eye characteristics and viewing position, ensuring accurate tracking regardless of eye shape, glasses, or sitting position.

### Benefits
- **Higher Accuracy**: Precision mapped to your specific eye movement range
- **Better Edge Detection**: Accurately tracks when you're looking at screen edges
- **Personalized Experience**: Adapts to your individual eye characteristics
- **Reduced Head Movement Dependence**: Less affected by small head movements
- **Visual Feedback**: Shows your eye position during calibration for better alignment

### Calibration Tips
- Ensure good, even lighting on your face
- Try to keep your head relatively still during calibration
- Look directly at each calibration point
- Complete all 5 points for the best accuracy
- Recalibrate if you significantly change your sitting position

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

## Recent Improvements

### Enhanced Display and Tracking
1. **Full HD Display**: Upgraded to 1920x1080 resolution for better visibility and more screen real estate.

2. **Pupil-Based Calibration**: Implemented a new calibration system that tracks actual pupil movement ranges.

3. **Improved Smoothing Algorithm**: Enhanced the temporal smoothing with an exponentially weighted system that gives higher priority to recent eye positions while still maintaining stability.

4. **Precise Edge Tracking**: Better tracking when looking at screen edges thanks to the calibrated range mapping.

5. **Centered Window Positioning**: Application windows now automatically center on your display.

6. **Visual Calibration Feedback**: Added real-time visualization of eye position during calibration.

### Visual Improvements
1. **Enhanced UI**: Cleaner, more intuitive interface with better contrast and readability.

2. **Pulsing Calibration Points**: Calibration points now pulse to draw attention and improve focus.

3. **Eye Position Visualization**: Shows the current position of your eyes during calibration for better feedback.

4. **Status Indicators**: Clear status messages with icons to guide the calibration process.

### Mac Compatibility
The `--mac-fix` flag continues to address common camera issues on Mac systems, particularly with Continuity Camera warnings.

## Analytics
The system tracks attention metrics for each defined region in the current layout:
- **Gaze Points**: The number of times a user looked at a region
- **Total Time**: The accumulated time spent looking at a region
- **Heatmap Overlay**: Visualizes attention intensity across different UI elements

Analytics can be saved to CSV files for later analysis, along with heatmap images.

## Troubleshooting

### Calibration Issues

If calibration seems problematic:

1. **Ensure Good, Even Lighting**: Calibration works best with diffuse lighting that doesn't cast shadows on your face or create glare on your eyes.

2. **Maintain Distance**: Sit approximately 50-70cm from the camera during calibration.

3. **Hold Still**: Try to keep your head still while calibrating each point.

4. **Complete All Points**: For best results, complete all 5 calibration points.

5. **Restart Calibration**: If you moved significantly during calibration, press ESC and restart the program to recalibrate.

### Eye Tracking Issues

If tracking still seems inaccurate after calibration:

1. **Recalibrate**: If you've moved or lighting has changed, recalibrate the system.

2. **Adjust Position**: Make sure you're seated at approximately the same distance as during calibration.

3. **Check Lighting**: Ensure lighting conditions haven't changed dramatically since calibration.

4. **Avoid Glasses Glare**: If wearing glasses, try adjusting the angle to reduce reflections.

5. **Debug Mode**: Run with `--debug` to see detailed eye tracking information.

### Mac-Specific Issues

If you're experiencing camera warning messages on Mac:

1. **Use the Mac fix**: Add the `--mac-fix` flag:
   ```bash
   python src/main.py --mac-fix
   ```

2. **Camera Permissions**: Ensure your application has permission to access the camera in System Preferences > Security & Privacy > Camera.

### Performance Issues

If the application is running slow:

1. **Close Other Applications**: Close resource-intensive applications that might compete for GPU or camera access.

2. **Disable Debug Output**: Make sure you're not running with the `--debug` flag.

### Contributors
- [Your Name] - [Contribution details]

## License
This project is for academic and research purposes.

---

# Quick Start Guide

For those who just want to get started quickly:

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run with calibration for best accuracy:
```bash
python src/main.py --mac-fix
```

3. Follow the on-screen calibration instructions, looking at each point and pressing SPACE.

4. Once calibrated, use keyboard controls:
   - Q: Quit
   - R: Reset heatmap
   - T: Toggle heatmap type
   - L: Change layout
   - H: Toggle help text

