import cv2
from face_detection import FaceDetector
from gaze_analysis import GazeAnalyzer

class EyeTracker:
    def __init__(self):
        self.face_detector = FaceDetector()
        self.gaze_analyzer = GazeAnalyzer()

    def process_frame(self, frame):
        # Detect faces and eyes
        faces = self.face_detector.detect_faces(frame)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            # Detect eyes within the face region
            roi_gray = frame[y:y+h, x:x+w]
            eyes = self.face_detector.eye_cascade.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(frame, (x+ex, y+ey), (x+ex+ew, y+ey+eh), (0, 255, 0), 2)

            # Update heatmap with detected eyes
            self.gaze_analyzer.update_heatmap(eyes)

        # Save results periodically
        if self.gaze_analyzer.frame_count % 30 == 0:  # Save every 30 frames
            self.gaze_analyzer.save_results() 