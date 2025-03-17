import cv2
from face_detection import FaceDetector
from gaze_analysis import GazeAnalyzer

class EyeTracker:
    def __init__(self):
        self.face_detector = FaceDetector()
        self.gaze_analyzer = GazeAnalyzer()
        self.frame_count = 0
        self.reset_interval = 300  # Reset heatmap every 300 frames (about 10 seconds at 30fps)

    def process_frame(self, frame):
        self.frame_count += 1
        
        # Reset heatmap periodically to prevent saturation
        if self.frame_count % self.reset_interval == 0:
            self.gaze_analyzer.reset_heatmap()
            print("Heatmap reset")
        
        # Detect faces and eyes
        faces = self.face_detector.detect_faces(frame)
        
        # If no faces detected, return early
        if len(faces) == 0:
            return
            
        # Process the largest face (closest to camera)
        largest_face = max(faces, key=lambda face: face[2] * face[3])
        (x, y, w, h) = largest_face
        
        # Draw face rectangle
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Detect eyes within the face region
        roi_gray = frame[y:y+h, x:x+w]
        eyes = self.face_detector.detect_eyes(roi_gray)
        
        # Convert eye coordinates to be relative to the full frame
        frame_relative_eyes = []
        for (ex, ey, ew, eh) in eyes:
            # Add the face position to get absolute coordinates
            abs_ex = x + ex
            abs_ey = y + ey
            
            # Draw eye rectangle
            cv2.rectangle(frame, (abs_ex, abs_ey), (abs_ex+ew, abs_ey+eh), (0, 255, 0), 2)
            
            # Add to list of eyes
            frame_relative_eyes.append((abs_ex, abs_ey, ew, eh))

        # Update heatmap with detected eyes (using frame-relative coordinates)
        if frame_relative_eyes:
            # Pass the frame and face info for gaze estimation
            self.gaze_analyzer.update_heatmap(
                frame_relative_eyes, 
                frame_shape=frame.shape, 
                frame=frame,
                face=(x, y, w, h)
            )
            
            # Collect gaze points from all eyes
            gaze_points = []
            for (ex, ey, ew, eh) in frame_relative_eyes:
                # Estimate gaze point
                gaze_x, gaze_y = self.gaze_analyzer.estimate_gaze_point(
                    ex, ey, ew, eh, frame, x, y, w, h
                )
                gaze_points.append((gaze_x, gaze_y, ex + ew // 2, ey + eh // 2))  # (gaze_x, gaze_y, eye_center_x, eye_center_y)
                
                # Draw a line from eye center to estimated gaze direction
                eye_center_x = ex + ew // 2
                eye_center_y = ey + eh // 2
                
                # Scale the gaze coordinates to fit on the camera frame
                scaled_gaze_x = int(gaze_x * frame.shape[1] / self.gaze_analyzer.screen_width)
                scaled_gaze_y = int(gaze_y * frame.shape[0] / self.gaze_analyzer.screen_height)
                
                # Draw a line showing individual eye gaze direction (thin line)
                cv2.line(frame, (eye_center_x, eye_center_y), (scaled_gaze_x, scaled_gaze_y), (255, 0, 0), 1)
            
            # If we have two eyes, calculate and visualize the average gaze point
            if len(gaze_points) == 2:
                # Calculate average gaze point
                avg_gaze_x = (gaze_points[0][0] + gaze_points[1][0]) // 2
                avg_gaze_y = (gaze_points[0][1] + gaze_points[1][1]) // 2
                
                # Calculate average eye center
                avg_eye_x = (gaze_points[0][2] + gaze_points[1][2]) // 2
                avg_eye_y = (gaze_points[0][3] + gaze_points[1][3]) // 2
                
                # Scale the average gaze coordinates to fit on the camera frame
                scaled_avg_gaze_x = int(avg_gaze_x * frame.shape[1] / self.gaze_analyzer.screen_width)
                scaled_avg_gaze_y = int(avg_gaze_y * frame.shape[0] / self.gaze_analyzer.screen_height)
                
                # Draw a thicker line showing the average gaze direction
                cv2.line(frame, (avg_eye_x, avg_eye_y), (scaled_avg_gaze_x, scaled_avg_gaze_y), (0, 0, 255), 2)
                
                # Add text showing the estimated screen coordinates
                cv2.putText(frame, f"Gaze: ({avg_gaze_x}, {avg_gaze_y})", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Save results periodically
        if self.frame_count % 30 == 0:  # Save every 30 frames
            self.gaze_analyzer.save_results() 