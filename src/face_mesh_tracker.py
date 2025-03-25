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
        
        # Head pose estimation variables - using more points for stability
        self.model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -63.6, -12.5),         # Chin
            (-43.3, 32.7, -26.0),        # Left eye left corner
            (43.3, 32.7, -26.0),         # Right eye right corner
            (-28.9, -28.9, -24.1),       # Left mouth corner
            (28.9, -28.9, -24.1),        # Right mouth corner
            (-34.0, 32.7, -26.0),        # Left eye right corner
            (34.0, 32.7, -26.0),         # Right eye left corner
        ], dtype=np.float32)
        
        # Camera matrix estimation (will be updated with actual dimensions)
        self.camera_matrix = None
        self.dist_coeffs = np.zeros((4, 1), dtype=np.float32)
        
        # Calibration variables
        self.is_calibrated = False
        self.calibration_points = []
        self.calibration_data = []
        
        # Store image dimensions
        self.image_width = None
        self.image_height = None
        
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
        height, width = frame.shape[:2]
        
        # Update stored dimensions
        self.image_height = height
        self.image_width = width
        
        # Convert to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Initialize camera matrix if not set
        if self.camera_matrix is None:
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
        
        try:
            # Extract landmarks as numpy array with pixel coordinates
            landmarks = []
            for landmark in face_landmarks.landmark:
                x = int(landmark.x * width)
                y = int(landmark.y * height)
                z = landmark.z * width  # Use width for z to maintain aspect ratio
                landmarks.append([x, y, z])
            landmarks = np.array(landmarks, dtype=np.float32)
            
            # Extract iris landmarks with pixel coordinates
            iris_landmarks = {
                'left': [],
                'right': []
            }
            
            for idx in self.LEFT_IRIS_INDICES:
                if idx < len(face_landmarks.landmark):
                    x = int(face_landmarks.landmark[idx].x * width)
                    y = int(face_landmarks.landmark[idx].y * height)
                    iris_landmarks['left'].append([x, y])
                    
            for idx in self.RIGHT_IRIS_INDICES:
                if idx < len(face_landmarks.landmark):
                    x = int(face_landmarks.landmark[idx].x * width)
                    y = int(face_landmarks.landmark[idx].y * height)
                    iris_landmarks['right'].append([x, y])
                    
            iris_landmarks['left'] = np.array(iris_landmarks['left'], dtype=np.float32)
            iris_landmarks['right'] = np.array(iris_landmarks['right'], dtype=np.float32)
            
            # Estimate head pose
            head_pose = self._estimate_head_pose(landmarks, (height, width))
            
            return landmarks, iris_landmarks, head_pose
            
        except Exception as e:
            print(f"Error processing landmarks: {str(e)}")
            return None, None, None
    
    def calibrate(self, frame, point):
        """
        Calibrate the eye tracker with a known point on screen.
        
        Args:
            frame: Current frame
            point: (x, y) coordinates of the calibration point on screen
            
        Returns:
            bool: True if calibration was successful
        """
        try:
            # Process frame to get landmarks
            landmarks, iris_landmarks, head_pose = self.process_frame(frame)
            
            if landmarks is None:
                print("No face detected")
                return False
                
            if iris_landmarks is None or len(iris_landmarks['left']) == 0 or len(iris_landmarks['right']) == 0:
                print("No iris landmarks detected")
                return False
                
            if head_pose is None:
                print("Could not estimate head pose")
                return False
            
            # Calculate gaze point
            gaze_point = self.calculate_normalized_gaze(iris_landmarks, head_pose)
            
            if gaze_point is None:
                print("Could not calculate gaze point")
                return False
                
            # Convert screen point to normalized coordinates
            height, width = frame.shape[:2]
            normalized_point = (point[0] / width, point[1] / height)
            
            # Store calibration data
            self.calibration_points.append(normalized_point)
            self.calibration_data.append(gaze_point)
            
            print(f"Calibration data recorded: Gaze ({gaze_point[0]:.3f}, {gaze_point[1]:.3f}) -> Target ({normalized_point[0]:.3f}, {normalized_point[1]:.3f})")
            
            # If we have enough calibration points, calculate transformation
            if len(self.calibration_points) >= 4:
                success = self._calculate_calibration_transform()
                if success:
                    self.is_calibrated = True
                    print("Calibration matrix calculated successfully")
                    return True
                else:
                    print("Failed to calculate calibration matrix")
                    return False
                    
            return True
            
        except Exception as e:
            print(f"Error during calibration: {str(e)}")
            return False
            
    def _calculate_calibration_transform(self):
        """
        Calculate the transformation matrix from normalized gaze to screen coordinates.
        
        Returns:
            bool: True if calculation was successful
        """
        try:
            if len(self.calibration_points) < 4:
                return False
                
            # Convert to numpy arrays
            src = np.array(self.calibration_data, dtype=np.float32)
            dst = np.array(self.calibration_points, dtype=np.float32)
            
            # Reshape for findHomography
            src = src.reshape(-1, 1, 2)
            dst = dst.reshape(-1, 1, 2)
            
            # Calculate transformation matrix with RANSAC for robustness
            self.calibration_matrix, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
            
            if self.calibration_matrix is None:
                return False
                
            # Check if the transformation is reasonable
            if np.any(np.isnan(self.calibration_matrix)) or np.any(np.isinf(self.calibration_matrix)):
                return False
                
            return True
            
        except Exception as e:
            print(f"Error calculating calibration transform: {str(e)}")
            return False
    
    def _estimate_head_pose(self, landmarks, frame_shape):
        """
        Estimate head pose using facial landmarks.
        
        Args:
            landmarks: Array of facial landmarks
            frame_shape: Shape of the input frame (height, width)
            
        Returns:
            (rotation_vector, translation_vector) of head pose
        """
        try:
            # Define the indices for stable facial landmarks
            landmark_indices = [
                1,    # Nose tip
                152,  # Chin
                33,   # Left eye left corner
                263,  # Right eye right corner
                61,   # Left mouth corner
                291,  # Right mouth corner
                133,  # Left eye right corner
                362,  # Right eye left corner
            ]
            
            # Get image points from landmarks
            image_points = []
            for idx in landmark_indices:
                if idx < len(landmarks):
                    point = landmarks[idx][:2]  # Only take x, y coordinates
                    if np.all(np.isfinite(point)):  # Check for valid points
                        image_points.append(point)
            
            image_points = np.array(image_points, dtype=np.float32)
            
            # Ensure we have enough valid points
            if len(image_points) < 6:
                return None
                
            # Use corresponding 3D model points
            model_points = self.model_points[:len(image_points)]
            
            # Solve for pose
            success, rotation_vector, translation_vector = cv2.solvePnP(
                model_points,
                image_points,
                self.camera_matrix,
                self.dist_coeffs,
                flags=cv2.SOLVEPNP_EPNP
            )
            
            if not success:
                return None
                
            return rotation_vector, translation_vector
            
        except Exception as e:
            print(f"Error in head pose estimation: {str(e)}")
            return None
    
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
            if len(iris_landmarks['left']) > 0:
                for point in iris_landmarks['left']:
                    cv2.circle(
                        annotated_frame,
                        (int(point[0]), int(point[1])),
                        1, (255, 0, 0), -1
                    )
            
            # Draw right iris
            if len(iris_landmarks['right']) > 0:
                for point in iris_landmarks['right']:
                    cv2.circle(
                        annotated_frame,
                        (int(point[0]), int(point[1])),
                        1, (255, 0, 0), -1
                    )
        
        # Draw head pose if available and valid
        if head_pose is not None and landmarks is not None:
            rotation_vector, translation_vector = head_pose
            
            try:
                # Draw axis only if we have valid rotation and translation
                if (rotation_vector is not None and translation_vector is not None and 
                    np.all(np.isfinite(rotation_vector)) and np.all(np.isfinite(translation_vector))):
                    
                    # Draw axis
                    axis_length = 50
                    points_3D = np.float32([
                        [0, 0, 0],         # Origin
                        [axis_length, 0, 0], # X-axis
                        [0, axis_length, 0], # Y-axis
                        [0, 0, axis_length]  # Z-axis
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
            except Exception as e:
                print(f"Warning: Could not draw head pose: {str(e)}")
        
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
        if iris_landmarks is None or len(iris_landmarks['left']) == 0 or len(iris_landmarks['right']) == 0:
            return None
            
        try:
            # Calculate center of each iris
            left_iris_center = np.mean(iris_landmarks['left'], axis=0)
            right_iris_center = np.mean(iris_landmarks['right'], axis=0)
            
            # Get the midpoint between eyes as reference
            eye_midpoint = (left_iris_center + right_iris_center) / 2
            
            # Calculate eye vector (direction the eyes are pointing)
            eye_vector = right_iris_center - left_iris_center
            eye_vector = eye_vector / np.linalg.norm(eye_vector)
            
            # Apply head pose correction if available
            if head_pose is not None:
                rotation_vector, translation_vector = head_pose
                
                # Convert rotation vector to rotation matrix
                rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
                
                # Get head angles
                euler_angles = self._rotation_matrix_to_euler_angles(rotation_matrix)
                pitch, yaw, roll = euler_angles
                
                # Create gaze direction vector considering head pose
                # The base vector points straight ahead
                gaze_vector = np.array([0, 0, -1])
                
                # Rotate gaze vector according to head pose
                gaze_vector = rotation_matrix.dot(gaze_vector)
                
                # Project the rotated vector onto the screen plane
                screen_point = eye_midpoint + gaze_vector * 500  # Project 500 units forward
                
                # Get the camera matrix dimensions
                if self.camera_matrix is not None:
                    fx = self.camera_matrix[0, 0]
                    fy = self.camera_matrix[1, 1]
                    cx = self.camera_matrix[0, 2]
                    cy = self.camera_matrix[1, 2]
                    
                    # Normalize coordinates based on camera parameters
                    normalized_x = (screen_point[0] - cx) / fx
                    normalized_y = (screen_point[1] - cy) / fy
                else:
                    # Fallback to simple normalization if no camera matrix
                    normalized_x = screen_point[0] / self.camera_matrix[0, 2] / 2
                    normalized_y = screen_point[1] / self.camera_matrix[1, 2] / 2
                
                # Adjust for head rotation
                normalized_x += yaw * 0.5
                normalized_y += pitch * 0.5
            else:
                # Fallback to simple iris center if no head pose
                if self.camera_matrix is not None:
                    normalized_x = (eye_midpoint[0] - self.camera_matrix[0, 2]) / self.camera_matrix[0, 0]
                    normalized_y = (eye_midpoint[1] - self.camera_matrix[1, 2]) / self.camera_matrix[1, 1]
                else:
                    normalized_x = eye_midpoint[0] / self.camera_matrix[0, 2] / 2
                    normalized_y = eye_midpoint[1] / self.camera_matrix[1, 2] / 2
            
            # Ensure values are between 0 and 1
            normalized_x = (normalized_x + 1) / 2  # Convert from [-1, 1] to [0, 1]
            normalized_y = (normalized_y + 1) / 2
            
            normalized_x = max(0, min(1, normalized_x))
            normalized_y = max(0, min(1, normalized_y))
            
            return normalized_x, normalized_y
            
        except Exception as e:
            print(f"Error in gaze calculation: {str(e)}")
            return None
    
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