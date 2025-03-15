import cv2
from eye_tracker import EyeTracker
import numpy as np

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

def main(video_source=0):
    # Initialize the eye tracker
    tracker = EyeTracker()
    
    # Create a simulated screen
    screen_width = 1920
    screen_height = 1080
    simulated_screen = create_test_screen(screen_width, screen_height)
    
    # Set the screen dimensions in the gaze analyzer
    tracker.gaze_analyzer.set_screen_dimensions(screen_width, screen_height)

    # Open the video source
    cap = cv2.VideoCapture(video_source)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process the frame
        tracker.process_frame(frame)

        # Display the frame with frame count
        frame_with_info = frame.copy()
        cv2.putText(frame_with_info, f"Frame: {tracker.frame_count}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Add info about heatmap status
        if tracker.gaze_analyzer.heatmap is not None:
            max_val = np.max(tracker.gaze_analyzer.heatmap)
            cv2.putText(frame_with_info, f"Heatmap max: {max_val:.2f}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.imshow('Eye Tracking', frame_with_info)
        
        # Display the simulated screen with gaze heatmap
        if tracker.gaze_analyzer.heatmap is not None:
            # Check if heatmap has any data
            if np.max(tracker.gaze_analyzer.heatmap) > 0:
                # Normalize the heatmap for display
                heatmap_normalized = cv2.normalize(tracker.gaze_analyzer.heatmap, None, 0, 255, cv2.NORM_MINMAX)
                heatmap_display = cv2.applyColorMap(heatmap_normalized.astype(np.uint8), cv2.COLORMAP_JET)
                
                # Resize heatmap to match screen size
                if simulated_screen.shape[:2] != heatmap_display.shape[:2]:
                    heatmap_display = cv2.resize(heatmap_display, 
                                               (simulated_screen.shape[1], simulated_screen.shape[0]))
                
                # Create a blended view of the screen and heatmap
                screen_with_heatmap = cv2.addWeighted(simulated_screen, 0.7, heatmap_display, 0.3, 0)
                
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

        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import sys
    video_source = sys.argv[1] if len(sys.argv) > 1 else 0
    main(video_source) 