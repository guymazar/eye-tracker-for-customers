import cv2
from eye_tracker import EyeTracker
import numpy as np
import argparse
from screen_layout import ScreenLayoutManager
import time
import os
import sys
import platform

# Fixed screen dimensions - no longer configurable via command line
SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080

def center_window(window_name, width, height):
    """Center the OpenCV window on the screen"""
    # Get screen dimensions - this is a bit of a hack but works on most systems
    try:
        # Try to use platform-specific approaches
        if platform.system() == "Windows":
            from win32api import GetSystemMetrics
            screen_width = GetSystemMetrics(0)
            screen_height = GetSystemMetrics(1)
        elif platform.system() == "Darwin":  # macOS
            import subprocess
            screen_info = subprocess.check_output(['system_profiler', 'SPDisplaysDataType']).decode('utf-8')
            for line in screen_info.split('\n'):
                if 'Resolution' in line:
                    parts = line.split(':')[1].strip().split(' x ')
                    if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                        screen_width = int(parts[0])
                        screen_height = int(parts[1])
                        break
            else:  # Default if not found
                screen_width = 1920
                screen_height = 1080
        else:  # Linux and others
            screen_width = 1920  # Default fallback
            screen_height = 1080
    except:
        # Fallback to common resolution if methods above fail
        screen_width = 1920
        screen_height = 1080
    
    # Calculate window position
    pos_x = (screen_width - width) // 2
    pos_y = (screen_height - height) // 2
    
    # Set window position
    cv2.moveWindow(window_name, pos_x, pos_y)
    print(f"Centering window at ({pos_x}, {pos_y})")

# Define a function to ensure a window is properly created and sized
def create_sized_window(window_name, width, height, fullscreen=True):
    """Create an OpenCV window with the specified size and ensure it's properly displayed"""
    if fullscreen:
        # Create a properly initialized fullscreen window
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    else:
        # Create a normal window with specified size
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, width, height)
    
    # Force the window to be at the specified size by drawing a dummy frame
    dummy_frame = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(dummy_frame, f"Initializing {window_name}...", (width//4, height//2),
               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
    cv2.imshow(window_name, dummy_frame)
    cv2.waitKey(1)  # Give time for window to update
    
    # Center the window if not fullscreen
    if not fullscreen:
        center_window(window_name, width, height)
    return dummy_frame

def main():
    # Setup argument parser
    parser = argparse.ArgumentParser(description='Eye Tracking for Customer Attention Analysis')
    parser.add_argument('--video', type=str, default=None, 
                        help='Path to video file (default: use webcam)')
    parser.add_argument('--traditional', action='store_true',
                        help='Use traditional eye tracking instead of MediaPipe')
    parser.add_argument('--temporal', action='store_true',
                        help='Use temporal heatmap by default (shows recent focus)')
    parser.add_argument('--auto-reset', type=int, default=30,
                        help='Auto-reset heatmap after this many seconds (0 to disable)')
    parser.add_argument('--layout', type=str, default='default',
                        help='Initial screen layout (default, grid, product_comparison, checkout)')
    parser.add_argument('--skip-calibration', action='store_true',
                        help='Skip the calibration phase')
    parser.add_argument('--simplified', action='store_true',
                        help='Run in simplified mode for more reliable operation (implies --traditional and --skip-calibration)')
    parser.add_argument('--debug', action='store_true',
                        help='Enable detailed debug output')
    parser.add_argument('--mac-fix', action='store_true',
                        help='Apply Mac-specific fixes for camera issues')
    
    args = parser.parse_args()
    
    # If simplified mode is enabled, override other settings
    if args.simplified:
        args.traditional = True
        args.skip_calibration = True
        print("Running in simplified mode with traditional tracking and no calibration")
    
    # Apply Mac-specific fixes if requested
    if args.mac_fix:
        import platform
        if platform.system() == "Darwin":
            # Suppress the Continuity Camera warning
            import os
            os.environ['OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS'] = '0'
            print("Applied Mac-specific camera fixes")
    
    # Initialize the eye tracker
    tracker = EyeTracker(use_mediapipe=not args.traditional, debug=True)
    
    # Create the screen layout manager with fixed dimensions
    screen_manager = ScreenLayoutManager(base_width=SCREEN_WIDTH, base_height=SCREEN_HEIGHT)
    
    # Set the initial layout
    simulated_screen = screen_manager.create_layout(args.layout)
    
    # Set the screen dimensions in the gaze analyzer
    screen_width = SCREEN_WIDTH  # Use the fixed dimensions
    screen_height = SCREEN_HEIGHT
    tracker.gaze_analyzer.set_screen_dimensions(screen_width, screen_height)
    
    # Print screen dimensions information
    print(f"Using fixed screen dimensions: {screen_width}x{screen_height}")
    
    # Configure heatmap settings
    tracker.gaze_analyzer.auto_reset_interval = args.auto_reset
    
    # Track which heatmap to display (temporal or cumulative)
    show_temporal = args.temporal
    
    # Track whether to show analytics overlay
    show_analytics = False
    
    # Add a flag to track if we need to end the session
    end_session = False
    
    # Track if we're using fallback mode due to tracking issues
    using_fallback = False
    tracking_error_count = 0
    max_errors_before_fallback = 50  # After this many consecutive errors, try fallback mode
    
    # Create a persistent cumulative heatmap for the entire session
    session_heatmap = None
    
    # Help text for the keyboard controls
    help_text = [
        "Keyboard Controls:",
        "  Q: Quit",
        "  R: Reset heatmap",
        "  r: Generate analytics report",
        "  T: Toggle between temporal and cumulative heatmap",
        "  H: Toggle help text",
        "  L: Switch to next layout",
        "  A: Toggle analytics overlay",
        "  S: Save current analytics",
        "  Click 'Stop Session' button to end and save results"
    ]
    show_help = True

    # Open the video source
    video_source = 0 if args.video is None else args.video
    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        print(f"Error: Could not open video source {video_source}")
        return

    # Print info about tracking method
    if not args.traditional:
        print("Using MediaPipe Face Mesh for improved eye tracking and head pose estimation")
    else:
        print("Using traditional eye tracking method")
    
    print(f"Using {'temporal' if show_temporal else 'cumulative'} heatmap for display")
    print(f"Initial layout: {args.layout}")
    print(f"Auto-reset interval: {args.auto_reset} seconds (0 = disabled)")
    print("Press 'h' to toggle help text")

    # Calibration phase
    if not args.skip_calibration:
        print("\nStarting pupil calibration phase...")
        print("Please follow the calibration points with your eyes.")
        print("For each point, look directly at it and press SPACE when ready.")
        print("Press ESC to skip calibration entirely.")
        
        # Define calibration corners with proper names
        calibration_points = [
            ("center", (screen_width // 2, screen_height // 2)),              # Center
            ("top_left", (screen_width // 8, screen_height // 8)),            # Top left
            ("top_right", (7 * screen_width // 8, screen_height // 8)),       # Top right
            ("bottom_left", (screen_width // 8, 7 * screen_height // 8)),     # Bottom left
            ("bottom_right", (7 * screen_width // 8, 7 * screen_height // 8)) # Bottom right
        ]
        
        current_point = 0
        
        # Create calibration window in fullscreen mode
        dummy_frame = create_sized_window('Calibration Feed', screen_width, screen_height, fullscreen=True)
        
        # Start the pupil calibration process
        if tracker.use_mediapipe:
            tracker.face_mesh_tracker.start_pupil_calibration()
        
        # Countdown variables
        countdown = 0
        countdown_started = False
        
        # Main calibration loop
        while current_point < len(calibration_points) and not args.skip_calibration:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process frame to get landmarks and draw them
            landmarks, iris_landmarks, head_pose = tracker.face_mesh_tracker.process_frame(frame)
            frame_with_landmarks = tracker.face_mesh_tracker.draw_landmarks(frame, landmarks, iris_landmarks, head_pose)
            
            # Get current calibration point info
            current_corner_name, current_point_pos = calibration_points[current_point]
            
            # Create a copy of the frame for drawing calibration overlay
            display_frame = frame_with_landmarks.copy()
            
            # Create a clean base for the overlay
            overlay = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
            
            # Draw grid lines (lighter)
            for i in range(0, screen_width, 100):
                cv2.line(overlay, (i, 0), (i, screen_height), (30, 30, 30), 1)
            for i in range(0, screen_height, 100):
                cv2.line(overlay, (0, i), (screen_width, i), (30, 30, 30), 1)
            
            # Draw all calibration points (dimmed)
            for i, (corner_name, point) in enumerate(calibration_points):
                if i == current_point:
                    # Current point is bright red with pulsing effect
                    pulse_size = 30 + int(10 * np.sin(time.time() * 5))  # Pulsating effect
                    cv2.circle(overlay, point, pulse_size, (0, 0, 255), -1)  # Red circle
                    cv2.circle(overlay, point, pulse_size + 5, (255, 255, 255), 2)  # White border
                else:
                    # Other points are gray
                    cv2.circle(overlay, point, 15, (80, 80, 80), -1)
            
            # Add text instructions with better visibility
            cv2.putText(overlay, f"Calibration Point {current_point + 1}/{len(calibration_points)}: {current_corner_name.replace('_', ' ').title()}", 
                       (screen_width // 4, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
            
            if countdown_started:
                if countdown > 0:
                    cv2.putText(overlay, f"Hold still... recording in {countdown}", (screen_width // 4, 120),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
                else:
                    cv2.putText(overlay, "Recording pupil position...", (screen_width // 4, 120),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
            else:
                cv2.putText(overlay, "Look at the RED circle and press SPACE when ready", (screen_width // 4, 120),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
            
            # Add eye tracking status with better formatting
            status_y_pos = 180
            if landmarks is None:
                cv2.putText(overlay, "⚠️ No face detected - please center your face in the camera", 
                           (screen_width // 4, status_y_pos), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            elif iris_landmarks is None or len(iris_landmarks['left']) == 0 or len(iris_landmarks['right']) == 0:
                cv2.putText(overlay, "⚠️ Eyes not detected - please ensure your eyes are visible", 
                           (screen_width // 4, status_y_pos), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                cv2.putText(overlay, "✅ Eyes tracked successfully", 
                           (screen_width // 4, status_y_pos), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Draw current eye position
                # Use a consistent fill color (not affected by pulsing)
                if iris_landmarks is not None:
                    left_center = np.mean(iris_landmarks['left'], axis=0).astype(int)
                    right_center = np.mean(iris_landmarks['right'], axis=0).astype(int)
                    
                    # Draw miniature eye tracking visualization
                    minimap_x = 50
                    minimap_y = screen_height - 150
                    minimap_width = 200
                    minimap_height = 100
                    
                    # Draw minimap background
                    cv2.rectangle(overlay, (minimap_x, minimap_y), 
                                 (minimap_x + minimap_width, minimap_y + minimap_height), 
                                 (40, 40, 40), -1)
                    cv2.rectangle(overlay, (minimap_x, minimap_y), 
                                 (minimap_x + minimap_width, minimap_y + minimap_height), 
                                 (100, 100, 100), 1)
                    
                    # Map eye positions to minimap
                    map_left_x = minimap_x + int((left_center[0] / frame.shape[1]) * minimap_width)
                    map_left_y = minimap_y + int((left_center[1] / frame.shape[0]) * minimap_height)
                    map_right_x = minimap_x + int((right_center[0] / frame.shape[1]) * minimap_width)
                    map_right_y = minimap_y + int((right_center[1] / frame.shape[0]) * minimap_height)
                    
                    # Draw eye positions
                    cv2.circle(overlay, (map_left_x, map_left_y), 5, (0, 255, 255), -1)
                    cv2.circle(overlay, (map_right_x, map_right_y), 5, (0, 255, 255), -1)
                    
                    # Label
                    cv2.putText(overlay, "Eye Tracking", (minimap_x, minimap_y - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # Add ESC to skip instruction
            cv2.putText(overlay, "Press ESC to skip calibration", (screen_width // 4, screen_height - 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150, 150, 150), 1)
            
            # Add this line before line 317 (before the blending code)
            opacity = 0.7  # Define opacity for blending (70% frame, 30% overlay)
            
            # Before blending, ensure overlay has the same dimensions as frame_with_landmarks
            if overlay.shape != frame_with_landmarks.shape:
                overlay = cv2.resize(overlay, (frame_with_landmarks.shape[1], frame_with_landmarks.shape[0]))
                
                # If channel count is different, convert overlay to match frame_with_landmarks
                if len(overlay.shape) != len(frame_with_landmarks.shape):
                    if len(frame_with_landmarks.shape) == 3:  # Color image
                        overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR)
                    else:  # Grayscale image
                        overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2GRAY)
            
            display_frame = cv2.addWeighted(frame_with_landmarks, opacity, overlay, 1 - opacity, 0)
            
            # Show the combined frame
            cv2.imshow('Calibration Feed', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' '):  # Space key
                if not countdown_started and landmarks is not None and iris_landmarks is not None:
                    # Start the automatic calibration process for this point
                    countdown_started = True
                    countdown = 3
                    print(f"Starting countdown for {current_corner_name}...")
                    countdown_time = time.time()  # Record when the countdown started
            
            # Handle automatic countdown and recording
            if countdown_started:
                current_time = time.time()
                
                # Calculate elapsed time since countdown started
                if countdown > 0:
                    # Check if 1 second has passed to decrement the countdown
                    if current_time - countdown_time >= 1.0:
                        countdown -= 1
                        countdown_time = current_time  # Reset the timer
                        print(f"Countdown: {countdown}")
                
                # When countdown reaches zero, record automatically
                elif countdown == 0:
                    try:
                        # Record pupil position for this corner
                        success = tracker.face_mesh_tracker.calibrate_pupil_at_corner(current_corner_name, iris_landmarks)
                        if success:
                            print(f"Calibration for {current_corner_name} successful!")
                            current_point += 1  # Automatically advance to next point
                            countdown_started = False
                            
                            # Add a short delay before moving to the next point
                            time.sleep(0.5)
                        else:
                            print(f"Calibration for {current_corner_name} failed. Please try again.")
                            countdown_started = False
                    except Exception as e:
                        print(f"Error during calibration: {str(e)}")
                        countdown_started = False
            
            elif key == 27:  # ESC key
                print("Calibration skipped")
                break
        
        cv2.destroyWindow('Calibration Feed')
        
        if current_point < len(calibration_points):
            print("Warning: Calibration was not completed. Results may be less accurate.")
        else:
            print("Calibration completed successfully!")
            print("You can now look around the screen and your gaze should be tracked accurately.")
            
    # Wait a moment to ensure windows are properly closed
    time.sleep(0.5)

    # Create and configure the main windows with proper sizes
    # Camera feed window - keep this at a normal size for monitoring
    create_sized_window('Eye Tracking', 640, 480, fullscreen=False)
    
    # Full-size window for the heatmap display - make this fullscreen
    dummy_frame = create_sized_window('Screen with Gaze Heatmap', screen_width, screen_height, fullscreen=True)
    
    # Define stop button dimensions and position
    button_width = 150
    button_height = 40
    button_x = 640 - button_width - 10 
    button_y = 10
    button_color = (0, 0, 150)
    button_hover_color = (0, 0, 200)
    button_pressed_color = (0, 0, 100)
    button_text = "Stop Session"
    button_state = "normal"  # normal, hover, pressed
    
    # Track mouse position and state
    mouse_pos = (0, 0)
    mouse_pressed = False
    
    # Function to check if mouse is over button
    def is_mouse_over_button(mx, my):
        return (button_x <= mx <= button_x + button_width and 
                button_y <= my <= button_y + button_height)
    
    # Mouse callback function
    def mouse_callback(event, x, y, flags, param):
        nonlocal mouse_pos, mouse_pressed, button_state, end_session
        
        mouse_pos = (x, y)
        
        # Check button state
        if is_mouse_over_button(x, y):
            if event == cv2.EVENT_LBUTTONDOWN:
                button_state = "pressed"
                mouse_pressed = True
            elif event == cv2.EVENT_LBUTTONUP:
                if button_state == "pressed":
                    # Button click completed, end the session
                    end_session = True
                button_state = "hover"
                mouse_pressed = False
            else:
                button_state = "hover"
        else:
            button_state = "normal"
            
        if event == cv2.EVENT_LBUTTONUP:
            mouse_pressed = False
    
    # Register mouse callback
    cv2.setMouseCallback('Eye Tracking', mouse_callback)

    while not end_session:
        ret, frame = cap.read()
        if not ret:
            break

        # Process the frame
        tracker.process_frame(frame)
        
        # Check if we're getting gaze data
        last_gaze_x, last_gaze_y = None, None
        
        if tracker.use_mediapipe and not using_fallback:
            # For MediaPipe version
            if hasattr(tracker, 'face_mesh_tracker') and hasattr(tracker.face_mesh_tracker, 'calculate_normalized_gaze'):
                # Get the latest processed frame data
                landmarks, iris_landmarks, head_pose = getattr(tracker, '_last_processed_data', (None, None, None))
                
                if iris_landmarks is not None and head_pose is not None:
                    normalized_gaze = tracker.face_mesh_tracker.calculate_normalized_gaze(iris_landmarks, head_pose)
                    if normalized_gaze is not None:
                        last_gaze_x = int(normalized_gaze[0] * screen_width)
                        last_gaze_y = int(normalized_gaze[1] * screen_height)
                        tracking_error_count = 0  # Reset error count when we get valid data
                    else:
                        tracking_error_count += 1
                else:
                    tracking_error_count += 1
            else:
                tracking_error_count += 1
                
            # If we've had too many errors, switch to fallback mode
            if tracking_error_count > max_errors_before_fallback and not using_fallback:
                print("\nSwitching to fallback tracking mode due to persistent errors")
                using_fallback = True
                tracker.use_mediapipe = False
                # Re-create the tracker in traditional mode
                tracker = EyeTracker(use_mediapipe=False, debug=True)
                print("Now using traditional eye tracking as fallback")
        
        # If we're using fallback or traditional mode
        if not tracker.use_mediapipe or using_fallback:
            # For traditional version - try to get the last gaze points
            if hasattr(tracker, '_last_gaze_points') and getattr(tracker, '_last_gaze_points', None):
                gaze_points = getattr(tracker, '_last_gaze_points')
                if len(gaze_points) == 2:  # If we have two eyes
                    # Use the average gaze point
                    last_gaze_x = (gaze_points[0][0] + gaze_points[1][0]) // 2
                    last_gaze_y = (gaze_points[0][1] + gaze_points[1][1]) // 2
                elif len(gaze_points) == 1:  # If we have one eye
                    last_gaze_x, last_gaze_y = gaze_points[0]
        
        # Update region analytics if we have a gaze point
        if last_gaze_x is not None and last_gaze_y is not None:
            screen_manager.update_region_analytics(last_gaze_x, last_gaze_y)
        
        # Display the simulated screen with gaze heatmap
        if tracker.gaze_analyzer.heatmap is not None:
            # Update the session-long cumulative heatmap
            if session_heatmap is None:
                # Initialize session heatmap with same dimensions as regular heatmap
                session_heatmap = np.zeros_like(tracker.gaze_analyzer.heatmap)
            
            # Add current heatmap to session heatmap (accumulate every point)
            session_heatmap += tracker.gaze_analyzer.heatmap * 0.01  # Scale to avoid overflow
            
            # Determine which screen to use as a base
            if show_analytics:
                screen_base = screen_manager.get_analytics_overlay()
            else:
                screen_base = screen_manager.current_screen
                
            # Get the appropriate heatmap based on current mode
            # For temporal/regular cumulative use the regular function
            if show_temporal:
                heatmap_display = tracker.gaze_analyzer.get_visualization_heatmap(use_temporal=True)
            else:
                # Use the persistent session heatmap for true cumulative view
                heatmap_display = cv2.applyColorMap(
                    cv2.normalize(session_heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
                    cv2.COLORMAP_JET
                )
            
            if heatmap_display is not None:
                # Create a blended view of the screen and heatmap
                opacity = 0.7  # Define opacity for blending (70% frame, 30% overlay)
                screen_with_heatmap = cv2.addWeighted(screen_base, opacity, heatmap_display, 1 - opacity, 0)
                
                # Add mode indicator to the heatmap display
                tracking_mode = "Traditional (Fallback)" if using_fallback else "MediaPipe" if tracker.use_mediapipe else "Traditional"
                heatmap_title = f"{('Temporal' if show_temporal else 'Cumulative')} Heatmap - {screen_manager.current_layout.capitalize()} - {tracking_mode}"
                cv2.putText(screen_with_heatmap, heatmap_title, (30, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                # Draw a marker at the current gaze position if available
                if last_gaze_x is not None and last_gaze_y is not None:
                    cv2.circle(screen_with_heatmap, (last_gaze_x, last_gaze_y), 15, (0, 0, 255), -1)
                    cv2.circle(screen_with_heatmap, (last_gaze_x, last_gaze_y), 15, (255, 255, 255), 2)
                
                # Show the screen at its native resolution (no longer resizing)
                cv2.imshow('Screen with Gaze Heatmap', screen_with_heatmap)
            else:
                # Just show the simulated screen if no heatmap data
                cv2.imshow('Screen with Gaze Heatmap', screen_base)

        # Display the frame with frame count and info
        frame_with_info = frame.copy()
        
        # Draw the stop button
        if button_state == "normal":
            cv2.rectangle(frame_with_info, (button_x, button_y), 
                         (button_x + button_width, button_y + button_height), 
                         button_color, -1)
        elif button_state == "hover":
            cv2.rectangle(frame_with_info, (button_x, button_y), 
                         (button_x + button_width, button_y + button_height), 
                         button_hover_color, -1)
        else:  # pressed
            cv2.rectangle(frame_with_info, (button_x, button_y), 
                         (button_x + button_width, button_y + button_height), 
                         button_pressed_color, -1)
        
        # Add button text
        cv2.putText(frame_with_info, button_text, 
                   (button_x + 10, button_y + button_height - 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add frame info
        cv2.putText(frame_with_info, f"Frame: {tracker.frame_count}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Add info about heatmap status
        if tracker.gaze_analyzer.heatmap is not None:
            # Show type of heatmap being displayed
            heatmap_type = "Temporal" if show_temporal else "Cumulative"
            cv2.putText(frame_with_info, f"{heatmap_type} Heatmap", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Show max value
            if show_temporal and tracker.gaze_analyzer.temporal_heatmap is not None:
                max_val = np.max(tracker.gaze_analyzer.temporal_heatmap)
            else:
                max_val = np.max(tracker.gaze_analyzer.heatmap)
                
            cv2.putText(frame_with_info, f"Max: {max_val:.2f}", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Show layout info
            cv2.putText(frame_with_info, f"Layout: {screen_manager.current_layout}", (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Display help text if enabled
        if show_help:
            y_pos = 30
            for line in help_text:
                cv2.putText(frame_with_info, line, (frame.shape[1] - 400, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y_pos += 25
        
        cv2.imshow('Eye Tracking', frame_with_info)
        
        # Check for key presses
        key = cv2.waitKey(1) & 0xFF
        
        # Exit on 'q' key
        if key == ord('q') or end_session:
            print("\nGenerating final analytics report...")
            try:
                # Save the cumulative session heatmap
                if session_heatmap is not None:
                    session_heatmap_path = os.path.join("output", "full_session_heatmap.png")
                    
                    # Normalize and apply colormap
                    normalized = cv2.normalize(session_heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                    colored_heatmap = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
                    
                    # Overlay on the last screen
                    final_display = cv2.addWeighted(screen_base, 0.7, colored_heatmap, 0.3, 0)
                    cv2.putText(final_display, "Full Session Heatmap", (30, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    
                    cv2.imwrite(session_heatmap_path, final_display)
                    print(f"Full session heatmap saved to: {session_heatmap_path}")
                    
                    # Pass the session heatmap to the screen manager for analytics report
                    screen_manager.include_session_heatmap(session_heatmap)
                
                # Generate and save final analytics
                csv_path, img_path = screen_manager.save_analytics()
                
                # Display message about where to find the report
                print(f"\nAnalytics saved successfully!")
                print(f"CSV data: {csv_path}")
                print(f"Heatmap image: {img_path}")
                print(f"Check the output/screen_analytics directory for the comprehensive visual report.")
                
                # Optional: try to open the report image with default viewer
                try:
                    import subprocess
                    import platform
                    
                    # Get report path (assumes it's in the same directory as the heatmap)
                    report_path = img_path.replace('_heatmap_', '_report_')
                    
                    # Different open commands based on OS
                    if platform.system() == 'Darwin':  # macOS
                        subprocess.call(('open', report_path))
                    elif platform.system() == 'Windows':
                        os.startfile(report_path)
                    else:  # linux
                        subprocess.call(('xdg-open', report_path))
                        
                    print(f"Opening report: {report_path}")
                except Exception as e:
                    print(f"Could not automatically open report: {str(e)}")
                    print("Please open the report file manually from the output directory.")
            except Exception as e:
                print(f"Error generating analytics report: {str(e)}")
            
            break
            
        # Reset heatmap on 'R' key
        elif key == ord('R'):
            tracker.gaze_analyzer.reset_heatmap()
            print("Heatmap manually reset")
            
        # Generate report on 'r' key (lowercase)
        elif key == ord('r'):
            print("Generating analytics report...")
            try:
                csv_path, img_path = screen_manager.save_analytics()
                print(f"Analytics report generated! Saved to {csv_path} and {img_path}")
                print("Check the output/screen_analytics directory for the comprehensive visual report.")
            except Exception as e:
                print(f"Error generating report: {str(e)}")
            
        # Toggle between temporal and cumulative heatmap on 't' key
        elif key == ord('t'):
            show_temporal = not show_temporal
            print(f"Switched to {('temporal' if show_temporal else 'cumulative')} heatmap display")
            
        # Toggle help text on 'h' key
        elif key == ord('h'):
            show_help = not show_help
            
        # Switch to next layout on 'l' key
        elif key == ord('l'):
            simulated_screen = screen_manager.next_layout()
            print(f"Switched to layout: {screen_manager.current_layout}")
            
        # Toggle analytics overlay on 'a' key
        elif key == ord('a'):
            show_analytics = not show_analytics
            print(f"Analytics overlay {'enabled' if show_analytics else 'disabled'}")
            
        # Save analytics on 's' key
        elif key == ord('s'):
            csv_path, img_path = screen_manager.save_analytics()
            print(f"Saved analytics to {csv_path} and heatmap to {img_path}")

    # Save final analytics before exiting
    screen_manager.save_analytics()
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 