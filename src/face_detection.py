import cv2

class FaceDetector:
    def __init__(self):
        # Load pre-trained Haar cascades
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')  # Better for glasses

    def detect_faces(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Increase minNeighbors to reduce false positives
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        return faces
        
    def detect_eyes(self, roi_gray):
        # Less strict parameters for eye detection to ensure we detect eyes
        eyes = self.eye_cascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.1,
            minNeighbors=4,  # Reduced from 6 to 4
            minSize=(20, 20),  # Reduced from 25x25 to 20x20
            maxSize=(80, 80)
        )
        
        # If we detect more than 2 eyes, only keep the 2 most likely ones
        if len(eyes) > 2:
            # Sort by area (larger eyes are more likely to be real)
            eyes = sorted(eyes, key=lambda e: e[2] * e[3], reverse=True)[:2]
            
        return eyes 