import cv2
from eye_tracker import EyeTracker
import numpy as np
import argparse
from screen_layout import ScreenLayoutManager

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
    
    args = parser.parse_args()
    
    # Initialize the eye tracker
    tracker = EyeTracker(use_mediapipe=not args.traditional)
    
    # Create the screen layout manager and initialize the first layout
    screen_manager = ScreenLayoutManager()
    
    # Set the initial layout
    simulated_screen = screen_manager.create_layout(args.layout)
    
    # Set the screen dimensions in the gaze analyzer
    screen_width = screen_manager.base_width
    screen_height = screen_manager.base_height
    tracker.gaze_analyzer.set_screen_dimensions(screen_width, screen_height)
    
    # Configure heatmap settings
    tracker.gaze_analyzer.auto_reset_interval = args.auto_reset
    
    # Track which heatmap to display (temporal or cumulative)
    show_temporal = args.temporal
    
    # Track whether to show analytics overlay
    show_analytics = False
    
    # Help text for the keyboard controls
    help_text = [
        "Keyboard Controls:",
        "  Q: Quit",
        "  R: Reset heatmap",
        "  T: Toggle between temporal and cumulative heatmap",
        "  H: Toggle help text",
        "  L: Switch to next layout",
        "  A: Toggle analytics overlay",
        "  S: Save current analytics"
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
        print("\nStarting calibration phase...")
        print("Please look at each calibration point when it appears.")
        print("Press SPACE when you're looking at the point.")
        print("Press ESC to skip calibration.")
        
        calibration_points = [
            (screen_width // 4, screen_height // 4),    # Top left
            (3 * screen_width // 4, screen_height // 4), # Top right
            (screen_width // 4, 3 * screen_height // 4), # Bottom left
            (3 * screen_width // 4, 3 * screen_height // 4) # Bottom right
        ]
        
        current_point = 0
        
        # Create window with specific position
        cv2.namedWindow('Calibration Feed', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Calibration Feed', 1280, 720)
        
        # Countdown variables
        countdown = 0
        countdown_started = False
        
        while current_point < len(calibration_points):
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process frame to get landmarks and draw them
            landmarks, iris_landmarks, head_pose = tracker.face_mesh_tracker.process_frame(frame)
            frame_with_landmarks = tracker.face_mesh_tracker.draw_landmarks(frame, landmarks, iris_landmarks, head_pose)
            
            # Create a copy of the frame for drawing calibration overlay
            display_frame = frame_with_landmarks.copy()
            
            # Draw semi-transparent overlay grid
            overlay = display_frame.copy()
            for i in range(0, screen_width, 100):
                cv2.line(overlay, (i, 0), (i, screen_height), (200, 200, 200), 1)
            for i in range(0, screen_height, 100):
                cv2.line(overlay, (0, i), (screen_width, i), (200, 200, 200), 1)
            
            # Draw all calibration points (dimmed)
            for i, point in enumerate(calibration_points):
                if i == current_point:
                    # Current point is bright red
                    cv2.circle(overlay, point, 30, (0, 0, 255), -1)  # Red circle
                    cv2.circle(overlay, point, 35, (255, 255, 255), 2)  # White border
                else:
                    # Other points are gray
                    cv2.circle(overlay, point, 20, (128, 128, 128), -1)
            
            # Add text instructions
            cv2.putText(overlay, f"Point {current_point + 1}/4", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            if countdown_started:
                if countdown > 0:
                    cv2.putText(overlay, f"Recording in {countdown}...", (10, 70),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                else:
                    cv2.putText(overlay, "Recording...", (10, 70),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                cv2.putText(overlay, "Press SPACE when ready", (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Add eye tracking status
            if landmarks is None:
                cv2.putText(overlay, "No face detected", (10, 110),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            elif iris_landmarks is None or len(iris_landmarks['left']) == 0 or len(iris_landmarks['right']) == 0:
                cv2.putText(overlay, "No eyes detected", (10, 110),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                cv2.putText(overlay, "Eyes tracked", (10, 110),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Draw current gaze point if available
                gaze_point = tracker.face_mesh_tracker.calculate_normalized_gaze(iris_landmarks, head_pose)
                if gaze_point is not None:
                    gaze_x = int(gaze_point[0] * screen_width)
                    gaze_y = int(gaze_point[1] * screen_height)
                    cv2.circle(overlay, (gaze_x, gaze_y), 10, (0, 255, 255), -1)  # Yellow dot for gaze
            
            # Blend the overlay with the original frame
            alpha = 0.7
            display_frame = cv2.addWeighted(overlay, alpha, display_frame, 1 - alpha, 0)
            
            # Show the combined frame
            cv2.imshow('Calibration Feed', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' '):  # Space key
                if not countdown_started:
                    countdown_started = True
                    countdown = 3
                    print("Starting countdown...")
                elif countdown > 0:
                    countdown -= 1
                    print(f"Countdown: {countdown}")
                elif countdown == 0:  # When countdown is finished
                    try:
                        if tracker.face_mesh_tracker.calibrate(frame, calibration_points[current_point]):
                            current_point += 1
                            print(f"Calibration point {current_point} recorded successfully!")
                            countdown_started = False
                            countdown = 0
                        else:
                            print("Failed to record calibration point. Please try again.")
                            countdown_started = False
                            countdown = 0
                    except Exception as e:
                        print(f"Error during calibration: {str(e)}")
                        print("Please try again.")
                        countdown_started = False
                        countdown = 0
            elif key == 27:  # ESC key
                print("Calibration skipped")
                break
        
        cv2.destroyWindow('Calibration Feed')
        
        if current_point < len(calibration_points):
            print("Warning: Calibration was not completed. Results may be less accurate.")
        else:
            print("Calibration completed successfully!")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process the frame
        tracker.process_frame(frame)

        # Display the frame with frame count and info
        frame_with_info = frame.copy()
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
        
        # Get the last estimated gaze point if available
        last_gaze_x, last_gaze_y = None, None
        
        if tracker.use_mediapipe:
            # For MediaPipe version
            if hasattr(tracker, 'face_mesh_tracker') and hasattr(tracker.face_mesh_tracker, 'calculate_normalized_gaze'):
                # Get the latest processed frame data
                landmarks, iris_landmarks, head_pose = getattr(tracker, '_last_processed_data', (None, None, None))
                
                if iris_landmarks is not None and head_pose is not None:
                    normalized_gaze = tracker.face_mesh_tracker.calculate_normalized_gaze(iris_landmarks, head_pose)
                    if normalized_gaze is not None:
                        last_gaze_x = int(normalized_gaze[0] * screen_width)
                        last_gaze_y = int(normalized_gaze[1] * screen_height)
        else:
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
            # Determine which screen to use as a base
            if show_analytics:
                screen_base = screen_manager.get_analytics_overlay()
            else:
                screen_base = screen_manager.current_screen
                
            # Get the appropriate heatmap based on current mode
            heatmap_display = tracker.gaze_analyzer.get_visualization_heatmap(use_temporal=show_temporal)
            
            if heatmap_display is not None:
                # Create a blended view of the screen and heatmap
                screen_with_heatmap = cv2.addWeighted(screen_base, 0.7, heatmap_display, 0.3, 0)
                
                # Add mode indicator to the heatmap display
                heatmap_title = f"{('Temporal' if show_temporal else 'Cumulative')} Heatmap - {screen_manager.current_layout.capitalize()}"
                cv2.putText(screen_with_heatmap, heatmap_title, (30, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                # Draw a marker at the current gaze position if available
                if last_gaze_x is not None and last_gaze_y is not None:
                    cv2.circle(screen_with_heatmap, (last_gaze_x, last_gaze_y), 15, (0, 0, 255), -1)
                    cv2.circle(screen_with_heatmap, (last_gaze_x, last_gaze_y), 15, (255, 255, 255), 2)
                
                # Resize for display if too large
                display_width = min(screen_width, 1280)
                display_height = int(display_width * screen_height / screen_width)
                screen_with_heatmap_resized = cv2.resize(screen_with_heatmap, (display_width, display_height))
                
                # Display the screen with heatmap
                cv2.imshow('Screen with Gaze Heatmap', screen_with_heatmap_resized)
            else:
                # Just show the simulated screen if no heatmap data
                display_width = min(screen_width, 1280)
                display_height = int(display_width * screen_height / screen_width)
                screen_resized = cv2.resize(screen_base, (display_width, display_height))
                cv2.imshow('Screen with Gaze Heatmap', screen_resized)

        # Check for key presses
        key = cv2.waitKey(1) & 0xFF
        
        # Exit on 'q' key
        if key == ord('q'):
            break
            
        # Reset heatmap on 'r' key
        elif key == ord('r'):
            tracker.gaze_analyzer.reset_heatmap()
            print("Heatmap manually reset")
            
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