import cv2
import numpy as np
import os

class GazeAnalyzer:
    def __init__(self, output_dir='output'):
        self.output_dir = output_dir
        self.heatmap = None
        self.frame_count = 0
        self.frame_dimensions_set = False
        self.screen_width = 1920  # Default screen width in pixels
        self.screen_height = 1080  # Default screen height in pixels
        
        # Calibration factors - adjust these to improve accuracy
        self.x_scale_factor = 1.2  # Increase to make horizontal eye movements more sensitive
        self.y_scale_factor = 1.0  # Increase to make vertical eye movements more sensitive
        self.x_offset = 0  # Adjust if gaze is consistently off horizontally
        self.y_offset = 0  # Adjust if gaze is consistently off vertically

        # Ensure output directories exist
        os.makedirs(os.path.join(self.output_dir, 'gaze_heatmaps'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'logs'), exist_ok=True)

    def reset_heatmap(self):
        """Reset the heatmap to all zeros"""
        if self.heatmap is not None:
            self.heatmap.fill(0)
            print("Heatmap reset to zeros")
        
    def set_screen_dimensions(self, width, height):
        """Set the dimensions of the screen being viewed"""
        self.screen_width = width
        self.screen_height = height
        
    def estimate_gaze_point(self, eye_x, eye_y, eye_w, eye_h, frame, face_x, face_y, face_w, face_h):
        """
        Estimate where the person is looking based on eye position and face orientation
        Returns (x, y) coordinates on the screen
        """
        # Extract the eye region
        eye_region = frame[eye_y:eye_y+eye_h, eye_x:eye_x+eye_w]
        
        # Convert to grayscale
        if len(eye_region.shape) == 3:
            gray_eye = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
        else:
            gray_eye = eye_region
            
        # Apply better thresholding to isolate the pupil
        # Use adaptive thresholding instead of global thresholding
        if eye_w > 20 and eye_h > 20:  # Only if eye region is big enough
            blurred = cv2.GaussianBlur(gray_eye, (5, 5), 0)
            thresholded = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY_INV, 11, 2
            )
        else:
            # Fallback to simple thresholding for small regions
            _, thresholded = cv2.threshold(gray_eye, 50, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours to locate the pupil
        contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # If no contours found, use the center of the eye as fallback
        if not contours:
            rel_pupil_x = 0.5
            rel_pupil_y = 0.5
        else:
            # Filter contours by size to avoid noise
            valid_contours = [c for c in contours if cv2.contourArea(c) > 5]
            
            if not valid_contours:
                rel_pupil_x = 0.5
                rel_pupil_y = 0.5
            else:
                # Find the largest contour (likely the pupil)
                pupil_contour = max(valid_contours, key=cv2.contourArea)
                
                # Get the center of the pupil
                M = cv2.moments(pupil_contour)
                if M["m00"] != 0:
                    pupil_x = int(M["m10"] / M["m00"])
                    pupil_y = int(M["m01"] / M["m00"])
                else:
                    pupil_x = eye_w // 2
                    pupil_y = eye_h // 2
                    
                # Calculate relative position of pupil within the eye (0-1)
                rel_pupil_x = pupil_x / eye_w
                rel_pupil_y = pupil_y / eye_h
        
        # Adjust for head position/orientation
        face_center_x = face_x + face_w // 2
        face_center_y = face_y + face_h // 2
        
        # Calculate head offset from center of frame
        frame_center_x = frame.shape[1] // 2
        frame_center_y = frame.shape[0] // 2
        
        head_offset_x = (face_center_x - frame_center_x) / frame.shape[1]
        head_offset_y = (face_center_y - frame_center_y) / frame.shape[0]
        
        # Map pupil position to screen coordinates with improved scaling
        # When pupil is to the left in the eye, person is looking right
        # When pupil is to the right in the eye, person is looking left
        # Apply scaling factors and offsets for better accuracy
        screen_x = int(((1 - rel_pupil_x) * self.screen_width * self.x_scale_factor - 
                       head_offset_x * 300) + self.x_offset)
        screen_y = int((rel_pupil_y * self.screen_height * self.y_scale_factor + 
                       head_offset_y * 150) + self.y_offset)
        
        # Ensure coordinates are within screen bounds
        screen_x = max(0, min(screen_x, self.screen_width - 1))
        screen_y = max(0, min(screen_y, self.screen_height - 1))
        
        return (screen_x, screen_y)
        
    def update_heatmap(self, eyes, frame_shape=None, frame=None, face=None):
        # Initialize heatmap based on screen dimensions, not camera dimensions
        if self.heatmap is None:
            self.heatmap = np.zeros((self.screen_height, self.screen_width), dtype=np.float32)
            print(f"Initialized heatmap with screen dimensions {self.screen_width}x{self.screen_height}")

        # Process all detected eyes
        if len(eyes) > 0 and frame is not None and face is not None:
            print(f"Processing {len(eyes)} detected eyes for gaze estimation")
            face_x, face_y, face_w, face_h = face
            
            # Collect gaze points from all eyes
            gaze_points = []
            for (ex, ey, ew, eh) in eyes:
                # Estimate where the person is looking
                gaze_x, gaze_y = self.estimate_gaze_point(ex, ey, ew, eh, frame, face_x, face_y, face_w, face_h)
                gaze_points.append((gaze_x, gaze_y))
            
            # If we have two eyes, use the average gaze point
            if len(gaze_points) == 2:
                gaze_x = (gaze_points[0][0] + gaze_points[1][0]) // 2
                gaze_y = (gaze_points[0][1] + gaze_points[1][1]) // 2
                
                # Update the heatmap at the average gaze point
                radius = 50  # Radius of influence in pixels
                
                # Define the region to update
                x_start = max(0, gaze_x - radius)
                y_start = max(0, gaze_y - radius)
                x_end = min(self.screen_width, gaze_x + radius)
                y_end = min(self.screen_height, gaze_y + radius)
                
                # Create a gaussian around the gaze point
                for y in range(y_start, y_end):
                    for x in range(x_start, x_end):
                        # Calculate distance from gaze point (squared)
                        dist_sq = (x - gaze_x)**2 + (y - gaze_y)**2
                        # Add more weight to center points with a gaussian
                        weight = np.exp(-dist_sq / (2 * (radius/3)**2)) * 5
                        self.heatmap[y, x] += weight
                
                # Print debug info
                print(f"Updated heatmap for average gaze point at screen coordinates ({gaze_x}, {gaze_y})")
                print(f"Current heatmap max value: {np.max(self.heatmap)}")
            else:
                # Otherwise, update for each eye individually
                for gaze_x, gaze_y in gaze_points:
                    # Update the heatmap at the gaze point
                    radius = 50  # Radius of influence in pixels
                    
                    # Define the region to update
                    x_start = max(0, gaze_x - radius)
                    y_start = max(0, gaze_y - radius)
                    x_end = min(self.screen_width, gaze_x + radius)
                    y_end = min(self.screen_height, gaze_y + radius)
                    
                    # Create a gaussian around the gaze point
                    for y in range(y_start, y_end):
                        for x in range(x_start, x_end):
                            # Calculate distance from gaze point (squared)
                            dist_sq = (x - gaze_x)**2 + (y - gaze_y)**2
                            # Add more weight to center points with a gaussian
                            weight = np.exp(-dist_sq / (2 * (radius/3)**2)) * 5
                            self.heatmap[y, x] += weight
                    
                    # Print debug info
                    print(f"Updated heatmap for gaze point at screen coordinates ({gaze_x}, {gaze_y})")
                    print(f"Current heatmap max value: {np.max(self.heatmap)}")

    def save_results(self):
        # Check if heatmap exists and has data
        if self.heatmap is None or np.max(self.heatmap) == 0:
            print("No gaze data collected yet, skipping heatmap generation")
            self.frame_count += 1
            return
            
        # Normalize the heatmap
        heatmap_normalized = cv2.normalize(self.heatmap, None, 0, 255, cv2.NORM_MINMAX)
        heatmap_colored = cv2.applyColorMap(heatmap_normalized.astype(np.uint8), cv2.COLORMAP_JET)

        # Save the heatmap
        heatmap_path = os.path.join(self.output_dir, 'gaze_heatmaps', f'heatmap_{self.frame_count}.png')
        cv2.imwrite(heatmap_path, heatmap_colored)

        # Log the heatmap path
        log_path = os.path.join(self.output_dir, 'logs', 'gaze_log.txt')
        with open(log_path, 'a') as log_file:
            log_file.write(f"Frame {self.frame_count}: {heatmap_path} (max value: {np.max(self.heatmap):.2f})\n")

        self.frame_count += 1 