import mediapipe as mp
import cv2
import numpy as np

class FaceMeshTracker:
    """
    Tracks facial landmarks using MediaPipe's Face Mesh solution.
    Provides precise eye landmarks and methods for better gaze estimation.
    """
    
    # Eye landmark indices (based on MediaPipe's 468-point face mesh)
    # Left eye landmarks
    LEFT_EYE_INDICES = [33, 133, 160, 159, 158, 144, 145, 153]
    LEFT_IRIS_INDICES = [474, 475, 476, 477]
    
    # Right eye landmarks
    RIGHT_EYE_INDICES = [362, 263, 386, 387, 388, 373, 374, 380]
    RIGHT_IRIS_INDICES = [469, 470, 471, 472]
    
    def __init__(self, static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        """
        Initialize the face mesh tracker.
        
        Args:
            static_image_mode: Whether to treat input as static images (vs video)
            max_num_faces: Maximum number of faces to detect
            min_detection_confidence: Minimum confidence for face detection
            min_tracking_confidence: Minimum confidence for face tracking
        """
        self.mp_face_mesh = mp.solutions.face_mesh
        
        # Initialize MediaPipe Face Mesh with iris landmarks
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=max_num_faces,
            refine_landmarks=True,  # Enable iris landmarks
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Head pose estimation variables
        self.model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye left corner
            (225.0, 170.0, -135.0),      # Right eye right corner
            (-150.0, -150.0, -125.0),    # Left mouth corner
            (150.0, -150.0, -125.0)      # Right mouth corner
        ])
        
        # Camera matrix estimation (will be updated with actual dimensions)
        self.camera_matrix = None
        self.dist_coeffs = np.zeros((4, 1))
        
    def process_frame(self, frame):
        """
        Process a frame to detect facial landmarks.
        
        Args:
            frame: Input BGR image
            
        Returns:
            landmarks: List of detected face landmarks if face is detected, None otherwise
            iris_landmarks: Dictionary with left and right iris landmarks
            head_pose: (rotation_vector, translation_vector) if face is detected, None otherwise
        """
        # Convert to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Initialize camera matrix if not set
        if self.camera_matrix is None:
            height, width, _ = frame.shape
            focal_length = width
            center = (width / 2, height / 2)
            self.camera_matrix = np.array(
                [[focal_length, 0, center[0]],
                 [0, focal_length, center[1]],
                 [0, 0, 1]], dtype=np.float64
            )
        
        # Process the image
        results = self.face_mesh.process(frame_rgb)
        
        if not results.multi_face_landmarks:
            return None, None, None
        
        # Get the first face's landmarks
        face_landmarks = results.multi_face_landmarks[0]
        
        # Extract landmarks as numpy array
        landmarks = np.array([
            [landmark.x * frame.shape[1], landmark.y * frame.shape[0], landmark.z * frame.shape[1]]
            for landmark in face_landmarks.landmark
        ])
        
        # Extract iris landmarks
        iris_landmarks = {
            'left': np.array([
                [face_landmarks.landmark[idx].x * frame.shape[1], 
                 face_landmarks.landmark[idx].y * frame.shape[0]]
                for idx in self.LEFT_IRIS_INDICES
            ]),
            'right': np.array([
                [face_landmarks.landmark[idx].x * frame.shape[1], 
                 face_landmarks.landmark[idx].y * frame.shape[0]]
                for idx in self.RIGHT_IRIS_INDICES
            ])
        }
        
        # Estimate head pose
        head_pose = self._estimate_head_pose(landmarks, frame.shape)
        
        return landmarks, iris_landmarks, head_pose
    
    def _estimate_head_pose(self, landmarks, frame_shape):
        """
        Estimate head pose using facial landmarks.
        
        Args:
            landmarks: Array of facial landmarks
            frame_shape: Shape of the input frame
            
        Returns:
            (rotation_vector, translation_vector) of head pose
        """
        # Get relevant facial landmarks for pose estimation
        # Indices are specific to MediaPipe Face Mesh
        image_points = np.array([
            landmarks[1],    # Nose tip
            landmarks[152],  # Chin
            landmarks[33],   # Left eye left corner
            landmarks[263],  # Right eye right corner
            landmarks[61],   # Left mouth corner
            landmarks[291]   # Right mouth corner
        ])[:, :2]  # Only take x and y coordinates
        
        # Solve for pose
        success, rotation_vector, translation_vector = cv2.solvePnP(
            self.model_points,
            image_points,
            self.camera_matrix,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if not success:
            return None
            
        return rotation_vector, translation_vector
    
    def draw_landmarks(self, frame, landmarks=None, iris_landmarks=None, head_pose=None):
        """
        Draw facial landmarks, iris landmarks, and head pose on the frame.
        
        Args:
            frame: Input BGR image
            landmarks: List of detected face landmarks
            iris_landmarks: Dictionary with left and right iris landmarks
            head_pose: (rotation_vector, translation_vector) of head pose
            
        Returns:
            Annotated frame
        """
        annotated_frame = frame.copy()
        
        # If we have face mesh results, draw them
        if landmarks is not None:
            # Draw face mesh landmarks (simplified)
            for idx, landmark in enumerate(landmarks):
                # Draw only eye landmarks for clarity
                if idx in self.LEFT_EYE_INDICES + self.RIGHT_EYE_INDICES:
                    cv2.circle(
                        annotated_frame,
                        (int(landmark[0]), int(landmark[1])),
                        1, (0, 255, 0), -1
                    )
        
        # Draw iris landmarks if available
        if iris_landmarks is not None:
            # Draw left iris
            for point in iris_landmarks['left']:
                cv2.circle(
                    annotated_frame,
                    (int(point[0]), int(point[1])),
                    1, (255, 0, 0), -1
                )
            
            # Draw right iris
            for point in iris_landmarks['right']:
                cv2.circle(
                    annotated_frame,
                    (int(point[0]), int(point[1])),
                    1, (255, 0, 0), -1
                )
        
        # Draw head pose if available
        if head_pose is not None:
            rotation_vector, translation_vector = head_pose
            
            # Draw axis
            axis_length = 50
            points_3D = np.array([
                [0, 0, 0],             # Origin
                [axis_length, 0, 0],   # X-axis
                [0, axis_length, 0],   # Y-axis
                [0, 0, axis_length]    # Z-axis
            ])
            
            # Project 3D points to image plane
            points_2D, _ = cv2.projectPoints(
                points_3D,
                rotation_vector,
                translation_vector,
                self.camera_matrix,
                self.dist_coeffs
            )
            
            # Convert to integers
            points_2D = np.int32(points_2D).reshape(-1, 2)
            
            # Draw axes
            origin = tuple(points_2D[0])
            cv2.line(annotated_frame, origin, tuple(points_2D[1]), (0, 0, 255), 2)  # X-axis (red)
            cv2.line(annotated_frame, origin, tuple(points_2D[2]), (0, 255, 0), 2)  # Y-axis (green)
            cv2.line(annotated_frame, origin, tuple(points_2D[3]), (255, 0, 0), 2)  # Z-axis (blue)
        
        return annotated_frame
    
    def calculate_normalized_gaze(self, iris_landmarks, head_pose):
        """
        Calculate normalized gaze coordinates considering head pose.
        
        Args:
            iris_landmarks: Dictionary with left and right iris landmarks
            head_pose: (rotation_vector, translation_vector) of head pose
            
        Returns:
            Normalized gaze coordinates (x, y) between 0 and 1
        """
        if iris_landmarks is None:
            return None
            
        # Calculate center of each iris
        left_iris_center = np.mean(iris_landmarks['left'], axis=0)
        right_iris_center = np.mean(iris_landmarks['right'], axis=0)
        
        # Average of both iris centers
        gaze_center = (left_iris_center + right_iris_center) / 2
        
        # Apply head pose correction if available
        if head_pose is not None:
            rotation_vector, translation_vector = head_pose
            
            # Convert rotation vector to rotation matrix
            rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
            
            # Get head angles
            euler_angles = self._rotation_matrix_to_euler_angles(rotation_matrix)
            pitch, yaw, roll = euler_angles
            
            # Adjust gaze based on head rotation
            # These values require calibration for best results
            correction_factor_x = 0.1
            correction_factor_y = 0.1
            
            # Corrected gaze
            gaze_center[0] -= yaw * correction_factor_x
            gaze_center[1] -= pitch * correction_factor_y
        
        # Normalize gaze coordinates to 0-1 range
        # Note: This assumes the camera's field of view covers the screen
        # For real applications, a calibration phase would map eye coordinates to screen coordinates
        normalized_x = max(0, min(1, gaze_center[0] / 1920))  # Assuming 1920 is screen width
        normalized_y = max(0, min(1, gaze_center[1] / 1080))  # Assuming 1080 is screen height
        
        return normalized_x, normalized_y
    
    def _rotation_matrix_to_euler_angles(self, R):
        """
        Convert rotation matrix to Euler angles (in radians).
        
        Args:
            R: 3x3 rotation matrix
            
        Returns:
            Euler angles (pitch, yaw, roll)
        """
        # Check if the rotation matrix is valid
        sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        
        if sy > 1e-6:  # Not singular
            x = np.arctan2(R[2, 1], R[2, 2])
            y = np.arctan2(-R[2, 0], sy)
            z = np.arctan2(R[1, 0], R[0, 0])
        else:  # Singular
            x = np.arctan2(-R[1, 2], R[1, 1])
            y = np.arctan2(-R[2, 0], sy)
            z = 0
            
        return np.array([x, y, z]) 