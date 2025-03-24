import cv2
from eye_tracker import EyeTracker
import numpy as np
import argparse

def create_test_screen(width=1920, height=1080):
    """Create a test screen with some UI elements to track gaze on"""
    screen = np.ones((height, width, 3), dtype=np.uint8) * 255  # White background
    
    # Add some UI elements
    # Header
    cv2.rectangle(screen, (0, 0), (width, 100), (200, 200, 200), -1)
    cv2.putText(screen, "Customer Attention Analysis", (width//2 - 200, 60), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2)
    
    # Left sidebar
    cv2.rectangle(screen, (0, 100), (300, height), (230, 230, 230), -1)
    
    # Main content area - add some product images or UI elements
    # Product 1
    cv2.rectangle(screen, (400, 200), (700, 500), (0, 0, 255), 2)
    cv2.putText(screen, "Product A", (500, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    # Product 2
    cv2.rectangle(screen, (800, 200), (1100, 500), (0, 255, 0), 2)
    cv2.putText(screen, "Product B", (900, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    # Product 3
    cv2.rectangle(screen, (1200, 200), (1500, 500), (255, 0, 0), 2)
    cv2.putText(screen, "Product C", (1300, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    # Bottom area - add some buttons
    cv2.rectangle(screen, (400, 600), (700, 700), (100, 100, 200), -1)
    cv2.putText(screen, "Buy Now", (500, 660), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    cv2.rectangle(screen, (800, 600), (1100, 700), (100, 200, 100), -1)
    cv2.putText(screen, "Add to Cart", (850, 660), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    cv2.rectangle(screen, (1200, 600), (1500, 700), (200, 100, 100), -1)
    cv2.putText(screen, "More Info", (1300, 660), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    return screen

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
    
    args = parser.parse_args()
    
    # Initialize the eye tracker
    tracker = EyeTracker(use_mediapipe=not args.traditional)
    
    # Create a simulated screen
    screen_width = 1920
    screen_height = 1080
    simulated_screen = create_test_screen(screen_width, screen_height)
    
    # Set the screen dimensions in the gaze analyzer
    tracker.gaze_analyzer.set_screen_dimensions(screen_width, screen_height)
    
    # Configure heatmap settings
    tracker.gaze_analyzer.auto_reset_interval = args.auto_reset
    
    # Track which heatmap to display (temporal or cumulative)
    show_temporal = args.temporal
    
    # Help text for the keyboard controls
    help_text = [
        "Keyboard Controls:",
        "  Q: Quit",
        "  R: Reset heatmap",
        "  T: Toggle between temporal and cumulative heatmap",
        "  H: Toggle help text"
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
    print(f"Auto-reset interval: {args.auto_reset} seconds (0 = disabled)")
    print("Press 'h' to toggle help text")

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
        
        # Display help text if enabled
        if show_help:
            y_pos = 30
            for line in help_text:
                cv2.putText(frame_with_info, line, (frame.shape[1] - 400, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y_pos += 25
        
        cv2.imshow('Eye Tracking', frame_with_info)
        
        # Display the simulated screen with gaze heatmap
        if tracker.gaze_analyzer.heatmap is not None:
            # Get the appropriate heatmap based on current mode
            heatmap_display = tracker.gaze_analyzer.get_visualization_heatmap(use_temporal=show_temporal)
            
            if heatmap_display is not None:
                # Create a blended view of the screen and heatmap
                screen_with_heatmap = cv2.addWeighted(simulated_screen, 0.7, heatmap_display, 0.3, 0)
                
                # Add mode indicator to the heatmap display
                heatmap_title = f"{('Temporal' if show_temporal else 'Cumulative')} Heatmap"
                cv2.putText(screen_with_heatmap, heatmap_title, (30, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
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
                simulated_screen_resized = cv2.resize(simulated_screen, (display_width, display_height))
                cv2.imshow('Screen with Gaze Heatmap', simulated_screen_resized)

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

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 