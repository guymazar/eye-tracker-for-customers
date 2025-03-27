import cv2
import numpy as np
import os
import time

class GazeAnalyzer:
    def __init__(self, output_dir='output'):
        self.output_dir = output_dir
        self.heatmap = None
        self.temporal_heatmap = None  # For tracking gaze points with time decay
        self.frame_count = 0
        self.frame_dimensions_set = False
        self.screen_width = 1920  # Default screen width in pixels
        self.screen_height = 1080  # Default screen height in pixels
        
        # Calibration factors - adjust these to improve accuracy
        self.x_scale_factor = 1.2  # Increase to make horizontal eye movements more sensitive
        self.y_scale_factor = 1.0  # Increase to make vertical eye movements more sensitive
        self.x_offset = 0  # Adjust if gaze is consistently off horizontally
        self.y_offset = 0  # Adjust if gaze is consistently off vertically
        
        # Heatmap configuration
        self.blur_radius = 25      # Size of Gaussian blur kernel
        self.decay_factor = 0.98   # Temporal decay factor (0.98 = 2% decay per frame)
        self.point_lifetime = 90   # Number of frames a point remains significant (at 30fps = ~3 seconds)
        self.last_reset_time = time.time()
        self.auto_reset_interval = 30  # Auto-reset heatmap after this many seconds (0 to disable)

        # Ensure output directories exist
        os.makedirs(os.path.join(self.output_dir, 'gaze_heatmaps'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'logs'), exist_ok=True)

    def reset_heatmap(self):
        """Reset the heatmap to all zeros"""
        if self.heatmap is not None:
            self.heatmap.fill(0)
            if self.temporal_heatmap is not None:
                self.temporal_heatmap.fill(0)
            self.last_reset_time = time.time()
            print("Heatmap reset to zeros")
        
    def set_screen_dimensions(self, width, height):
        """Set the dimensions of the screen being viewed"""
        self.screen_width = width
        self.screen_height = height
        
        # Initialize heatmaps
        if self.heatmap is None:
            self.heatmap = np.zeros((self.screen_height, self.screen_width), dtype=np.float32)
            self.temporal_heatmap = np.zeros((self.screen_height, self.screen_width), dtype=np.float32)
        
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
            
        # Apply improved eye processing
        if eye_w > 20 and eye_h > 20:  # Only if eye region is big enough
            # Apply histogram equalization to enhance pupil contrast
            gray_eye = cv2.equalizeHist(gray_eye)
            
            # Apply a bilateral filter to reduce noise while preserving edges
            gray_eye = cv2.bilateralFilter(gray_eye, 9, 75, 75)
            
            # Apply a Gaussian blur
            blurred = cv2.GaussianBlur(gray_eye, (7, 7), 0)
            
            # Use adaptive thresholding for more reliable pupil detection
            thresholded = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY_INV, 11, 2
            )
            
            # Apply morphology operations to clean up the threshold result
            kernel = np.ones((3, 3), np.uint8)
            thresholded = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, kernel, iterations=1)
            thresholded = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, kernel, iterations=2)
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
            # Filter contours by size to avoid noise and select the most likely pupil
            min_area = 5  # Minimum contour area to consider
            valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]
            
            if not valid_contours:
                rel_pupil_x = 0.5
                rel_pupil_y = 0.5
            else:
                # Find the most circular contour - likely to be the pupil
                best_pupil_contour = None
                best_circularity = 0
                
                for contour in valid_contours:
                    area = cv2.contourArea(contour)
                    perimeter = cv2.arcLength(contour, True)
                    if perimeter > 0:
                        # Circularity = 4π*area/perimeter²
                        circularity = 4 * np.pi * area / (perimeter * perimeter)
                        if circularity > best_circularity:
                            best_circularity = circularity
                            best_pupil_contour = contour
                
                # If we still don't have a good contour, use the largest one
                if best_pupil_contour is None:
                    best_pupil_contour = max(valid_contours, key=cv2.contourArea)
                
                # Get the center of the pupil
                M = cv2.moments(best_pupil_contour)
                if M["m00"] != 0:
                    pupil_x = int(M["m10"] / M["m00"])
                    pupil_y = int(M["m01"] / M["m00"])
                else:
                    pupil_x = eye_w // 2
                    pupil_y = eye_h // 2
                    
                # Calculate relative position of pupil within the eye (0-1)
                rel_pupil_x = pupil_x / eye_w
                rel_pupil_y = pupil_y / eye_h
        
        # Enhanced calibration factors for improved responsiveness
        # Increase these values to make eye tracking more sensitive
        # Current default values significantly increased to make eyes more responsive
        x_scale_factor = 1.5  # Reduced from 3.0 to 1.5 for less horizontal sensitivity
        y_scale_factor = 1.2  # Reduced from 2.5 to 1.2 for less vertical sensitivity
        
        # Adjust for head position/orientation
        face_center_x = face_x + face_w // 2
        face_center_y = face_y + face_h // 2
        
        # Calculate head offset from center of frame
        frame_center_x = frame.shape[1] // 2
        frame_center_y = frame.shape[0] // 2
        
        head_offset_x = (face_center_x - frame_center_x) / frame.shape[1]
        head_offset_y = (face_center_y - frame_center_y) / frame.shape[0]
        
        # Map pupil position to screen coordinates with enhanced sensitivity
        # When pupil is to the left in the eye, person is looking right
        # When pupil is to the right in the eye, person is looking left
        
        # Map relative pupil position to [-0.5, 0.5] range for easier manipulation
        centered_pupil_x = rel_pupil_x - 0.5
        centered_pupil_y = rel_pupil_y - 0.5
        
        # Apply non-linear transformation to enhance sensitivity but more gently
        # Using power 0.9 instead of 0.8 for more gradual response
        enhanced_pupil_x = np.sign(centered_pupil_x) * np.power(abs(centered_pupil_x), 0.9)
        enhanced_pupil_y = np.sign(centered_pupil_y) * np.power(abs(centered_pupil_y), 0.9)
        
        # Map back to [0, 1] range but with enhanced sensitivity
        enhanced_rel_pupil_x = 0.5 - enhanced_pupil_x
        enhanced_rel_pupil_y = 0.5 + enhanced_pupil_y
        
        # Add a deadzone in the center to reduce jitter
        # If the pupil is very close to center (within 5% of center), treat it as center
        deadzone = 0.05
        if abs(centered_pupil_x) < deadzone:
            enhanced_rel_pupil_x = 0.5
        if abs(centered_pupil_y) < deadzone:
            enhanced_rel_pupil_y = 0.5
        
        # Final mapping to screen coordinates with head position correction
        # Reduce the impact of head position by decreasing the multipliers
        screen_x = int(((enhanced_rel_pupil_x) * self.screen_width * x_scale_factor - 
                       head_offset_x * 150) + self.x_offset)  # Reduced from 300 to 150
        screen_y = int((enhanced_rel_pupil_y * self.screen_height * y_scale_factor + 
                       head_offset_y * 75) + self.y_offset)   # Reduced from 150 to 75
        
        # Apply exponential moving average for smoother movement
        # This adds some lag but reduces jitter significantly
        if hasattr(self, 'last_screen_x') and hasattr(self, 'last_screen_y'):
            # Blend with previous position (80% previous, 20% new)
            smoothing_factor = 0.8
            screen_x = int(smoothing_factor * self.last_screen_x + (1 - smoothing_factor) * screen_x)
            screen_y = int(smoothing_factor * self.last_screen_y + (1 - smoothing_factor) * screen_y)
        
        # Store current position for next frame
        self.last_screen_x = screen_x
        self.last_screen_y = screen_y
        
        # Ensure coordinates are within screen bounds
        screen_x = max(0, min(screen_x, self.screen_width - 1))
        screen_y = max(0, min(screen_y, self.screen_height - 1))
        
        return (screen_x, screen_y)
        
    def update_heatmap(self, eyes, frame_shape=None, frame=None, face=None):
        # Initialize heatmap based on screen dimensions, not camera dimensions
        if self.heatmap is None:
            self.heatmap = np.zeros((self.screen_height, self.screen_width), dtype=np.float32)
            self.temporal_heatmap = np.zeros((self.screen_height, self.screen_width), dtype=np.float32)
            print(f"Initialized heatmap with screen dimensions {self.screen_width}x{self.screen_height}")

        # Apply temporal decay to the temporal heatmap
        self._apply_temporal_decay()
        
        # Check for auto-reset
        self._check_auto_reset()

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
                self._update_heatmaps_with_point(gaze_x, gaze_y)
            else:
                # Otherwise, update for each eye individually
                for gaze_x, gaze_y in gaze_points:
                    self._update_heatmaps_with_point(gaze_x, gaze_y)

    def update_heatmap_with_point(self, gaze_x, gaze_y):
        """
        Update the heatmap with a single gaze point.
        This method is used by the MediaPipe face mesh implementation.
        
        Args:
            gaze_x: X-coordinate of the gaze point (in screen coordinates)
            gaze_y: Y-coordinate of the gaze point (in screen coordinates)
        """
        # Initialize heatmap if it doesn't exist
        if self.heatmap is None:
            self.heatmap = np.zeros((self.screen_height, self.screen_width), dtype=np.float32)
            self.temporal_heatmap = np.zeros((self.screen_height, self.screen_width), dtype=np.float32)
            print(f"Initialized heatmap with screen dimensions {self.screen_width}x{self.screen_height}")
        
        # Apply temporal decay to the temporal heatmap
        self._apply_temporal_decay()
        
        # Check for auto-reset
        self._check_auto_reset()
        
        # Update heatmaps with the gaze point
        self._update_heatmaps_with_point(gaze_x, gaze_y)
        
    def _update_heatmaps_with_point(self, gaze_x, gaze_y):
        """
        Internal method to update both cumulative and temporal heatmaps with a gaze point.
        
        Args:
            gaze_x: X-coordinate of the gaze point
            gaze_y: Y-coordinate of the gaze point
        """
        # Ensure coordinates are within screen bounds
        gaze_x = max(0, min(gaze_x, self.screen_width - 1))
        gaze_y = max(0, min(gaze_y, self.screen_height - 1))
        
        # Create a temporary heatmap for this gaze point
        single_point = np.zeros((self.screen_height, self.screen_width), dtype=np.float32)
        
        # Add a single point with high intensity
        single_point[gaze_y, gaze_x] = 255
        
        # Apply Gaussian blur for a more natural, smooth heatmap point
        # Adjust kernel size for different smooth levels
        blurred_point = cv2.GaussianBlur(single_point, (self.blur_radius, self.blur_radius), 0)
        
        # Normalize the blurred point to maintain consistent intensity
        if np.max(blurred_point) > 0:  # Avoid division by zero
            blurred_point = blurred_point * (255 / np.max(blurred_point)) * 0.2  # Scale factor for intensity
        
        # Add to both heatmaps
        self.heatmap += blurred_point
        self.temporal_heatmap += blurred_point
        
        # Print debug info
        print(f"Updated heatmap for gaze point at screen coordinates ({gaze_x}, {gaze_y})")
        print(f"Current heatmap max value: {np.max(self.heatmap):.2f}")
    
    def _apply_temporal_decay(self):
        """Apply temporal decay to the temporal heatmap"""
        if self.temporal_heatmap is not None and self.decay_factor < 1.0:
            self.temporal_heatmap *= self.decay_factor
    
    def _check_auto_reset(self):
        """Check if it's time for an automatic heatmap reset"""
        if self.auto_reset_interval > 0:
            current_time = time.time()
            if current_time - self.last_reset_time > self.auto_reset_interval:
                self.reset_heatmap()
                print(f"Auto-reset heatmap after {self.auto_reset_interval} seconds")
    
    def get_visualization_heatmap(self, use_temporal=True, colormap=cv2.COLORMAP_JET):
        """
        Generate a visualization-ready heatmap.
        
        Args:
            use_temporal: Whether to use the temporal heatmap (True) or cumulative heatmap (False)
            colormap: The OpenCV colormap to use
            
        Returns:
            Colored heatmap ready for visualization or overlay
        """
        # Use the temporal or cumulative heatmap
        source_heatmap = self.temporal_heatmap if use_temporal and self.temporal_heatmap is not None else self.heatmap
        
        if source_heatmap is None or np.max(source_heatmap) == 0:
            # Return an empty colored heatmap if there's no data
            empty = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)
            return empty
        
        # Normalize the heatmap to 0-255 range
        heatmap_normalized = cv2.normalize(source_heatmap, None, 0, 255, cv2.NORM_MINMAX)
        
        # Apply a final Gaussian blur for smoother appearance
        heatmap_normalized = cv2.GaussianBlur(heatmap_normalized, (5, 5), 0)
        
        # Convert to uint8 and apply colormap
        heatmap_colored = cv2.applyColorMap(heatmap_normalized.astype(np.uint8), colormap)
        
        return heatmap_colored

    def save_results(self):
        # Check if heatmap exists and has data
        if self.heatmap is None or np.max(self.heatmap) == 0:
            print("No gaze data collected yet, skipping heatmap generation")
            self.frame_count += 1
            return
            
        # Save both cumulative and temporal heatmaps
        cumulative_heatmap = self.get_visualization_heatmap(use_temporal=False)
        temporal_heatmap = self.get_visualization_heatmap(use_temporal=True)
        
        # Save the heatmaps
        cumulative_path = os.path.join(self.output_dir, 'gaze_heatmaps', f'cumulative_heatmap_{self.frame_count}.png')
        temporal_path = os.path.join(self.output_dir, 'gaze_heatmaps', f'temporal_heatmap_{self.frame_count}.png')
        
        cv2.imwrite(cumulative_path, cumulative_heatmap)
        cv2.imwrite(temporal_path, temporal_heatmap)

        # Log the heatmap paths
        log_path = os.path.join(self.output_dir, 'logs', 'gaze_log.txt')
        with open(log_path, 'a') as log_file:
            log_file.write(f"Frame {self.frame_count}:\n")
            log_file.write(f"  Cumulative: {cumulative_path} (max: {np.max(self.heatmap):.2f})\n")
            log_file.write(f"  Temporal: {temporal_path} (max: {np.max(self.temporal_heatmap):.2f})\n")
            log_file.write(f"  Time since reset: {time.time() - self.last_reset_time:.2f}s\n\n")

        self.frame_count += 1 