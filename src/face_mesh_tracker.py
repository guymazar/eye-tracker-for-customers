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
    
    def __init__(self, static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5, debug=False):
        """
        Initialize the face mesh tracker.
        
        Args:
            static_image_mode: Whether to treat input as static images (vs video)
            max_num_faces: Maximum number of faces to detect
            min_detection_confidence: Minimum confidence for face detection
            min_tracking_confidence: Minimum confidence for face tracking
            debug: Whether to print detailed debug information
        """
        self.mp_face_mesh = mp.solutions.face_mesh
        self.debug = debug
        
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
        
        # Pupil tracking calibration
        self.pupil_calibration_complete = False
        self.initial_left_pupil_pos = None
        self.initial_right_pupil_pos = None
        self.pupil_movement_range = {
            'left': {'min_x': None, 'max_x': None, 'min_y': None, 'max_y': None},
            'right': {'min_x': None, 'max_x': None, 'min_y': None, 'max_y': None}
        }
        self.calibration_corner_data = {}  # Will store pupil positions for each corner
        
        # Smoothing variables
        self.gaze_history = []
        self.history_length = 5  # Number of frames to keep for smoothing
        
        # Store image dimensions
        self.image_width = None
        self.image_height = None
        
        # Add a member variable to store landmarks
        self.landmarks = None
        
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
        
        # Process the image - pass frame directly (older API style)
        results = self.face_mesh.process(frame_rgb)
        
        if not results.multi_face_landmarks:
            self.landmarks = None  # Clear landmarks if no face detected
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
            
            # Store landmarks as a class member for use by other methods
            self.landmarks = landmarks
            
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
            self.landmarks = None  # Clear landmarks on error
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
    
    def start_pupil_calibration(self):
        """Initialize the pupil calibration process"""
        self.pupil_calibration_complete = False
        self.calibration_corner_data = {}
        print("Starting pupil calibration process")
        return True
        
    def calibrate_pupil_at_corner(self, corner_name, iris_landmarks):
        """
        Record pupil position when looking at a specific corner
        
        Args:
            corner_name: Name of the corner ('center', 'top_left', 'top_right', 'bottom_left', 'bottom_right')
            iris_landmarks: Dictionary with left and right iris landmarks
            
        Returns:
            bool: True if calibration for this point was successful
        """
        if iris_landmarks is None or len(iris_landmarks['left']) == 0 or len(iris_landmarks['right']) == 0:
            print(f"Cannot calibrate for {corner_name} - no iris landmarks detected")
            return False
            
        try:
            # Calculate the center of each iris
            left_iris_center = np.mean(iris_landmarks['left'], axis=0)
            right_iris_center = np.mean(iris_landmarks['right'], axis=0)
            
            # Store the data for this corner
            self.calibration_corner_data[corner_name] = {
                'left': left_iris_center.copy(),
                'right': right_iris_center.copy()
            }
            
            # If this is the center point, also save it as the initial pupil position
            if corner_name == 'center':
                self.initial_left_pupil_pos = left_iris_center.copy()
                self.initial_right_pupil_pos = right_iris_center.copy()
                print(f"Initial pupil positions recorded: Left {self.initial_left_pupil_pos}, Right {self.initial_right_pupil_pos}")
                
            # Check if we have all the required corners
            required_corners = ['center', 'top_left', 'top_right', 'bottom_left', 'bottom_right']
            if all(corner in self.calibration_corner_data for corner in required_corners):
                self._calculate_pupil_movement_range()
                self.pupil_calibration_complete = True
                print("Pupil calibration completed successfully!")
                
            return True
            
        except Exception as e:
            print(f"Error during pupil calibration at {corner_name}: {str(e)}")
            return False
            
    def _calculate_pupil_movement_range(self):
        """Calculate the min/max range of pupil movement based on calibration data"""
        try:
            # Get all x and y positions for each eye
            left_x_positions = [data['left'][0] for data in self.calibration_corner_data.values()]
            left_y_positions = [data['left'][1] for data in self.calibration_corner_data.values()]
            right_x_positions = [data['right'][0] for data in self.calibration_corner_data.values()]
            right_y_positions = [data['right'][1] for data in self.calibration_corner_data.values()]
            
            # Calculate min and max for each eye
            self.pupil_movement_range['left']['min_x'] = min(left_x_positions)
            self.pupil_movement_range['left']['max_x'] = max(left_x_positions)
            self.pupil_movement_range['left']['min_y'] = min(left_y_positions)
            self.pupil_movement_range['left']['max_y'] = max(left_y_positions)
            
            self.pupil_movement_range['right']['min_x'] = min(right_x_positions)
            self.pupil_movement_range['right']['max_x'] = max(right_x_positions)
            self.pupil_movement_range['right']['min_y'] = min(right_y_positions)
            self.pupil_movement_range['right']['max_y'] = max(right_y_positions)
            
            # Print the ranges
            print(f"Pupil movement ranges calculated:")
            print(f"Left eye X: {self.pupil_movement_range['left']['min_x']:.1f} to {self.pupil_movement_range['left']['max_x']:.1f}")
            print(f"Left eye Y: {self.pupil_movement_range['left']['min_y']:.1f} to {self.pupil_movement_range['left']['max_y']:.1f}")
            print(f"Right eye X: {self.pupil_movement_range['right']['min_x']:.1f} to {self.pupil_movement_range['right']['max_x']:.1f}")
            print(f"Right eye Y: {self.pupil_movement_range['right']['min_y']:.1f} to {self.pupil_movement_range['right']['max_y']:.1f}")
            
            return True
            
        except Exception as e:
            print(f"Error calculating pupil movement range: {str(e)}")
            return False
            
    def calculate_normalized_gaze(self, iris_landmarks, head_pose):
        """
        Calculate normalized gaze coordinates based on pupil position.
        If calibrated, uses the pupil movement ranges, otherwise falls back to simpler method.
        
        Args:
            iris_landmarks: Dictionary with left and right iris landmarks
            head_pose: (rotation_vector, translation_vector) of head pose
            
        Returns:
            Normalized gaze coordinates (x, y) between 0 and 1
        """
        if iris_landmarks is None or len(iris_landmarks['left']) == 0 or len(iris_landmarks['right']) == 0:
            if self.debug:
                print("Warning: Missing iris landmarks")
            return None
            
        try:
            # Check if landmarks are available
            if self.landmarks is None:
                if self.debug:
                    print("Warning: No face landmarks available for gaze calculation")
                return None
            
            # Get eye contour landmarks for reference
            left_eye_landmarks = []
            right_eye_landmarks = []
            
            for idx in self.LEFT_EYE_INDICES:
                if idx < len(self.landmarks):
                    left_eye_landmarks.append(self.landmarks[idx])
                elif self.debug:
                    print(f"Warning: Left eye landmark index {idx} out of range")
            
            for idx in self.RIGHT_EYE_INDICES:
                if idx < len(self.landmarks):
                    right_eye_landmarks.append(self.landmarks[idx])
                elif self.debug:
                    print(f"Warning: Right eye landmark index {idx} out of range")
            
            # Check if we have enough landmarks for both eyes
            if len(left_eye_landmarks) == 0 or len(right_eye_landmarks) == 0:
                if self.debug:
                    print(f"Warning: Insufficient eye landmarks - Left: {len(left_eye_landmarks)}, Right: {len(right_eye_landmarks)}")
                return None
            
            # Calculate centers of eye contours (not iris)
            left_eye_center = np.mean(left_eye_landmarks, axis=0)
            right_eye_center = np.mean(right_eye_landmarks, axis=0)
            
            # Calculate the center of each iris
            left_iris_center = np.mean(iris_landmarks['left'], axis=0)
            right_iris_center = np.mean(iris_landmarks['right'], axis=0)
            
            # Calculate pupil offset relative to eye center (this is what we emphasize)
            left_pupil_offset = None
            right_pupil_offset = None
            
            if left_eye_landmarks and right_eye_landmarks:
                try:
                    # Make sure dimensions match - eye landmarks may include z coordinate (3D) while iris landmarks are 2D
                    # Extract only x, y components from eye landmarks
                    left_eye_center_2d = left_eye_center[:2] if len(left_eye_center) > 2 else left_eye_center
                    right_eye_center_2d = right_eye_center[:2] if len(right_eye_center) > 2 else right_eye_center
                    
                    # Calculate eye width using only x, y components for consistency
                    left_eye_points_2d = [landmark[:2] for landmark in left_eye_landmarks]
                    right_eye_points_2d = [landmark[:2] for landmark in right_eye_landmarks]
                    
                    # Use first and midpoint landmarks for width calculation
                    left_eye_width = max(np.linalg.norm(np.array(left_eye_points_2d[0]) - np.array(left_eye_points_2d[4])), 1)
                    right_eye_width = max(np.linalg.norm(np.array(right_eye_points_2d[0]) - np.array(right_eye_points_2d[4])), 1)
                    
                    # Calculate pupil offset as a percentage of eye width
                    left_pupil_offset = (left_iris_center - left_eye_center_2d) / left_eye_width
                    right_pupil_offset = (right_iris_center - right_eye_center_2d) / right_eye_width
                    
                    if self.debug:
                        print(f"Pupil offsets calculated successfully - Left: {left_pupil_offset}, Right: {right_pupil_offset}")
                        
                except Exception as e:
                    if self.debug:
                        print(f"Error calculating pupil offsets: {str(e)}")
                    left_pupil_offset = None
                    right_pupil_offset = None
            elif self.debug:
                print("Warning: Could not calculate pupil offsets - eye landmarks missing")

            # Use calibrated pupil tracking if available
            if self.pupil_calibration_complete:
                # Verify calibration data
                if self.debug:
                    print(f"Using calibrated tracking with ranges: Left X: {self.pupil_movement_range['left']['min_x']:.1f}-{self.pupil_movement_range['left']['max_x']:.1f}, "
                          f"Y: {self.pupil_movement_range['left']['min_y']:.1f}-{self.pupil_movement_range['left']['max_y']:.1f}")
                    print(f"Current iris positions - Left: {left_iris_center}, Right: {right_iris_center}")
                
                try:
                    # Calculate the normalized position based on calibrated ranges
                    # For x, we need to invert the mapping (pupil moves left = look right)
                    left_x_norm = 1.0 - self._normalize_value(
                        left_iris_center[0],
                        self.pupil_movement_range['left']['min_x'],
                        self.pupil_movement_range['left']['max_x']
                    )
                    left_y_norm = self._normalize_value(
                        left_iris_center[1],
                        self.pupil_movement_range['left']['min_y'],
                        self.pupil_movement_range['left']['max_y']
                    )
                    
                    right_x_norm = 1.0 - self._normalize_value(
                        right_iris_center[0],
                        self.pupil_movement_range['right']['min_x'],
                        self.pupil_movement_range['right']['max_x']
                    )
                    right_y_norm = self._normalize_value(
                        right_iris_center[1],
                        self.pupil_movement_range['right']['min_y'],
                        self.pupil_movement_range['right']['max_y']
                    )
                    
                    # Enhanced pupil-centric tracking - incorporate relative pupil position
                    if left_pupil_offset is not None and right_pupil_offset is not None:
                        # Weight: Give 80% importance to calibrated pupil position, 20% to relative pupil offset
                        pupil_position_weight = 0.8
                        
                        # Ensure offset dimensions match expected values
                        left_offset_x = float(left_pupil_offset[0])
                        left_offset_y = float(left_pupil_offset[1])
                        right_offset_x = float(right_pupil_offset[0])
                        right_offset_y = float(right_pupil_offset[1])
                        
                        # Adjust normalized values based on pupil offset
                        # These adjustments fine-tune the gaze direction based on pupil position within the eye
                        left_x_norm = left_x_norm * pupil_position_weight + (0.5 - left_offset_x) * (1 - pupil_position_weight)
                        left_y_norm = left_y_norm * pupil_position_weight + (left_offset_y + 0.5) * (1 - pupil_position_weight)
                        
                        right_x_norm = right_x_norm * pupil_position_weight + (0.5 - right_offset_x) * (1 - pupil_position_weight)
                        right_y_norm = right_y_norm * pupil_position_weight + (right_offset_y + 0.5) * (1 - pupil_position_weight)
                    
                    # Average the normalized positions from both eyes
                    normalized_x = (left_x_norm + right_x_norm) / 2
                    normalized_y = (left_y_norm + right_y_norm) / 2
                    
                    if self.debug:
                        print(f"Calibrated gaze: Left ({left_x_norm:.2f}, {left_y_norm:.2f}), "
                              f"Right ({right_x_norm:.2f}, {right_y_norm:.2f}), "
                              f"Avg ({normalized_x:.2f}, {normalized_y:.2f})")
                        
                except Exception as e:
                    if self.debug:
                        print(f"Error in calibrated tracking calculation: {str(e)}")
                    # Fall back to simpler method
                    normalized_x = 0.5
                    normalized_y = 0.5
            else:
                # Fall back to the pupil-centric method if not calibrated
                try:
                    if left_pupil_offset is not None and right_pupil_offset is not None:
                        # Put 70% weight on pupil offset within the eye
                        pupil_offset_weight = 0.7
                        
                        # Ensure offset dimensions match expected values
                        left_offset_x = float(left_pupil_offset[0])
                        left_offset_y = float(left_pupil_offset[1])
                        right_offset_x = float(right_pupil_offset[0])
                        right_offset_y = float(right_pupil_offset[1])
                        
                        # Calculate normalized gaze from pupil offset
                        # When pupil is to the left in the eye, person is looking right (invert x)
                        pupil_x = 0.5 - (left_offset_x + right_offset_x) / 2
                        pupil_y = 0.5 + (left_offset_y + right_offset_y) / 2
                        
                        # Also consider absolute position for fallback (30% weight)
                        position_x = (float(left_iris_center[0]) + float(right_iris_center[0])) / (2 * self.image_width)
                        position_y = (float(left_iris_center[1]) + float(right_iris_center[1])) / (2 * self.image_height)
                        position_x = 1.0 - position_x  # Invert for natural eye movement mapping
                        
                        # Combine both factors with appropriate weights
                        normalized_x = pupil_x * pupil_offset_weight + position_x * (1 - pupil_offset_weight)
                        normalized_y = pupil_y * pupil_offset_weight + position_y * (1 - pupil_offset_weight)
                        
                        if self.debug:
                            print(f"Using pupil-centric method - Normalized: ({normalized_x:.2f}, {normalized_y:.2f})")
                    else:
                        # Fallback to basic position if we couldn't calculate offsets
                        normalized_x = 1.0 - (float(left_iris_center[0]) + float(right_iris_center[0])) / (2 * self.image_width)
                        normalized_y = (float(left_iris_center[1]) + float(right_iris_center[1])) / (2 * self.image_height)
                        
                        if self.debug:
                            print(f"Using basic position method - Normalized: ({normalized_x:.2f}, {normalized_y:.2f})")
                    
                    # Apply sensitivity adjustments
                    x_sensitivity = 1.1  # Reduced sensitivity for more stable tracking
                    y_sensitivity = 0.9
                    
                    # Center offset to map [0,1] space to [-0.5,0.5] space for manipulation
                    normalized_x = (normalized_x - 0.5) * x_sensitivity + 0.5
                    normalized_y = (normalized_y - 0.5) * y_sensitivity + 0.5
                except Exception as e:
                    if self.debug:
                        print(f"Error in uncalibrated tracking calculation: {str(e)}")
                    # Use middle of screen as fallback
                    normalized_x = 0.5
                    normalized_y = 0.5
            
            # Apply a deadzone in the center to reduce jitter
            deadzone = 0.03
            if abs(normalized_x - 0.5) < deadzone:
                normalized_x = 0.5
            if abs(normalized_y - 0.5) < deadzone:
                normalized_y = 0.5
            
            # Apply head pose correction if available, with reduced influence
            if head_pose is not None:
                rotation_vector, translation_vector = head_pose
                
                # Convert rotation vector to rotation matrix
                rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
                
                # Get head angles
                euler_angles = self._rotation_matrix_to_euler_angles(rotation_matrix)
                pitch, yaw, roll = euler_angles
                
                # Apply a gentler head pose correction (reduced from previous values)
                head_influence = 0.02  # Reduced to minimize head position influence
                normalized_x += yaw * head_influence
                normalized_y += pitch * head_influence
            
            # Apply exponential moving average for smoother movement
            if self.gaze_history:
                # Blend with previous positions (80% previous, 20% new)
                blend_factor = 0.8
                history_weight = blend_factor / len(self.gaze_history)
                
                # Calculate weighted average using recent history
                weighted_x = normalized_x * (1 - blend_factor)
                weighted_y = normalized_y * (1 - blend_factor)
                
                for x, y in self.gaze_history:
                    weighted_x += x * history_weight
                    weighted_y += y * history_weight
                
                normalized_x = weighted_x
                normalized_y = weighted_y
            
            # Update history
            self.gaze_history.append((normalized_x, normalized_y))
            if len(self.gaze_history) > self.history_length:
                self.gaze_history.pop(0)
            
            # Ensure values are in [0, 1] range
            normalized_x = max(0, min(1, normalized_x))
            normalized_y = max(0, min(1, normalized_y))
            
            return (normalized_x, normalized_y)
            
        except Exception as e:
            if self.debug:
                print(f"Error calculating normalized gaze: {str(e)}")
            return None
    
    def _normalize_value(self, value, min_val, max_val):
        """Normalize a value between 0 and 1 based on a min/max range"""
        # Avoid division by zero
        if max_val == min_val:
            return 0.5
            
        normalized = (value - min_val) / (max_val - min_val)
        return max(0, min(1, normalized))  # Clamp between 0 and 1
    
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