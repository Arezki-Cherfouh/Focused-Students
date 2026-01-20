import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def calculate_angle(a, b, c):
    """Calculate angle between three points"""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

def is_hand_raised(landmarks, image_width, image_height):
    """Detect if hand is raised above shoulder"""
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
    right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
    nose = landmarks[mp_pose.PoseLandmark.NOSE.value]
    
    left_hand_raised = (left_wrist.y < left_shoulder.y and 
                        left_wrist.y < nose.y + 0.1 and
                        left_wrist.visibility > 0.5)
    
    right_hand_raised = (right_wrist.y < right_shoulder.y and 
                         right_wrist.y < nose.y + 0.1 and
                         right_wrist.visibility > 0.5)
    
    return left_hand_raised or right_hand_raised

# def is_focused(landmarks):
#     """Determine if student is focused based on head orientation and posture"""
#     # Key landmarks
#     nose = landmarks[mp_pose.PoseLandmark.NOSE.value]
#     l_ear = landmarks[mp_pose.PoseLandmark.LEFT_EAR.value]
#     r_ear = landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value]
#     l_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
#     r_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    
#     # 1. DETECT SIDE TURN (Looking away)
#     # Measure horizontal distance from nose to each ear
#     dist_l = abs(nose.x - l_ear.x)
#     dist_r = abs(nose.x - r_ear.x)
    
#     # If the ratio is skewed (e.g., one distance is 3x larger than the other), head is turned
#     # A focused person has a ratio close to 1.0
#     head_turn_ratio = dist_l / (dist_r + 1e-6)
#     is_not_turned = 0.35 < head_turn_ratio < 2.8
    
#     # 2. DETECT LOOKING DOWN (Phone/Lap)
#     # Nose Y should be significantly smaller (higher up) than shoulder Y
#     avg_shoulder_y = (l_shoulder.y + r_shoulder.y) / 2
#     # If nose drops too close to shoulder level, they are looking down
#     # (The 0.08 offset is the "alert" zone)
#     is_upright = nose.y < (avg_shoulder_y - 0.08)
    
#     # 3. DETECT SLUMPING/SIDE TILT
#     # Ears should be relatively level
#     ear_level_diff = abs(l_ear.y - r_ear.y)
#     is_level = ear_level_diff < 0.05
    
#     return is_not_turned and is_upright and is_level

# def is_focused(landmarks):
#     """Determine if student is focused based on eye-to-ear alignment and symmetry"""
#     # Get key landmarks
#     nose = landmarks[mp_pose.PoseLandmark.NOSE.value]
#     l_eye = landmarks[mp_pose.PoseLandmark.LEFT_EYE.value]
#     r_eye = landmarks[mp_pose.PoseLandmark.RIGHT_EYE.value]
#     l_ear = landmarks[mp_pose.PoseLandmark.LEFT_EAR.value]
#     r_ear = landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value]
#     l_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
#     r_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]

#     # 1. DETECT LOOKING DOWN (Eye vs Ear Verticality)
#     # In a normal focused position, eyes are level with or higher than ears.
#     # If eye Y-coordinate becomes significantly larger (lower) than ear Y, they are looking down.
#     avg_eye_y = (l_eye.y + r_eye.y) / 2
#     avg_ear_y = (l_ear.y + r_ear.y) / 2
    
#     # Threshold 0.02: If eyes drop below the ear-line, they are looking down.
#     is_not_looking_down = avg_eye_y < (avg_ear_y + 0.02)

#     # 2. DETECT SIDE TURN (Eye Symmetry)
#     # We check if the nose is centered between the eyes. 
#     # This is more forgiving than using ears.
#     dist_l = abs(nose.x - l_eye.x)
#     dist_r = abs(nose.x - r_eye.x)
    
#     # Avoid division by zero and check ratio
#     eye_ratio = dist_l / (dist_r + 1e-6)
#     # A focused student will have a ratio between 0.5 and 2.0
#     is_facing_forward = 0.5 < eye_ratio < 2.0

#     # 3. DETECT SLUMPING (Nose vs Shoulder)
#     # Ensure the head hasn't dropped into the chest
#     avg_shoulder_y = (l_shoulder.y + r_shoulder.y) / 2
#     is_not_slumping = nose.y < (avg_shoulder_y - 0.05)

#     # All conditions must be met to be "Focused"
#     return is_not_looking_down and is_facing_forward and is_not_slumping

def is_focused(landmarks):
    """Determine if student is focused with improved sensitivity thresholds"""
    # Get key landmarks
    nose = landmarks[mp_pose.PoseLandmark.NOSE.value]
    l_eye = landmarks[mp_pose.PoseLandmark.LEFT_EYE.value]
    r_eye = landmarks[mp_pose.PoseLandmark.RIGHT_EYE.value]
    l_ear = landmarks[mp_pose.PoseLandmark.LEFT_EAR.value]
    r_ear = landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value]
    l_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    r_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]

    # 1. DETECT LOOKING DOWN (Eye vs Ear Verticality)
    # Sensitivity Fix: Lowered the threshold. If eyes are level with ears (or lower),
    # it indicates a downward head tilt toward a phone or book.
    avg_eye_y = (l_eye.y + r_eye.y) / 2
    avg_ear_y = (l_ear.y + r_ear.y) / 2
    
    # Using -0.01 instead of +0.02 makes it much faster to detect looking down.
    is_not_looking_down = avg_eye_y < (avg_ear_y - 0.01)

    # 2. DETECT SIDE TURN (Eye Symmetry)
    # Sensitivity Fix: Widened the ratio from (0.5 - 2.0) to (0.25 - 4.0).
    # This allows the student to look at the corners of their screen without being flagged.
    dist_l = abs(nose.x - l_eye.x)
    dist_r = abs(nose.x - r_eye.x)
    
    eye_ratio = dist_l / (dist_r + 1e-6)
    is_facing_forward = 0.25 < eye_ratio < 4.0

    # 3. DETECT SLUMPING (Nose vs Shoulder)
    avg_shoulder_y = (l_shoulder.y + r_shoulder.y) / 2
    is_not_slumping = nose.y < (avg_shoulder_y - 0.05)

    return is_not_looking_down and is_facing_forward and is_not_slumping

def get_bounding_box(landmarks, image_width, image_height):
    """Calculate bounding box around the person"""
    x_coords = []
    y_coords = []
    
    for landmark in landmarks:
        if landmark.visibility > 0.5:
            x_coords.append(int(landmark.x * image_width))
            y_coords.append(int(landmark.y * image_height))
    
    if x_coords and y_coords:
        x_min = max(0, min(x_coords) - 20)
        x_max = min(image_width, max(x_coords) + 20)
        y_min = max(0, min(y_coords) - 20)
        y_max = min(image_height, max(y_coords) + 20)
        return (x_min, y_min, x_max, y_max)
    
    return None

def main():
    cap = cv2.VideoCapture(0)
    print("Student Monitor Started - Press 'q' to quit")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1)
        image_height, image_width, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)
        
        student_count = 0
        
        if results.pose_landmarks:
            student_count = 1
            landmarks = results.pose_landmarks.landmark
            bbox = get_bounding_box(landmarks, image_width, image_height)
            
            if bbox:
                x_min, y_min, x_max, y_max = bbox
                
                # Logic Fix: If is_focused returns False, status becomes Distracted
                focused = is_focused(landmarks)
                color = (0, 255, 0) if focused else (0, 165, 255) 
                status = "Focused" if focused else "DISTRACTED"
                
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 3)
                label_y = y_min - 10 if y_min > 30 else y_max + 25
                cv2.putText(frame, status, (x_min, label_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                if is_hand_raised(landmarks, image_width, image_height):
                    hand_label_y = y_min - 40 if y_min > 60 else y_max + 55
                    cv2.putText(frame, "RAISED HAND", (x_min, hand_label_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    cv2.circle(frame, (x_min - 15, hand_label_y - 5), 10, (0, 0, 255), -1)
        
        cv2.putText(frame, f"Students: {student_count}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow('Student Focus Monitor', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
    pose.close()

if __name__ == "__main__":
    main()



# import cv2
# import mediapipe as mp
# import numpy as np

# # Initialize MediaPipe Pose
# mp_pose = mp.solutions.pose
# mp_drawing = mp.solutions.drawing_utils
# pose = mp_pose.Pose(
#     min_detection_confidence=0.5,
#     min_tracking_confidence=0.5
# )

# def calculate_angle(a, b, c):
#     """Calculate angle between three points"""
#     a = np.array(a)
#     b = np.array(b)
#     c = np.array(c)
    
#     radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
#     angle = np.abs(radians*180.0/np.pi)
    
#     if angle > 180.0:
#         angle = 360 - angle
        
#     return angle

# def is_hand_raised(landmarks, image_width, image_height):
#     """Detect if hand is raised above shoulder"""
#     # Get relevant landmarks
#     left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
#     right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
#     left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
#     right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
#     nose = landmarks[mp_pose.PoseLandmark.NOSE.value]
    
#     # Check if either wrist is above the shoulder and near head level
#     left_hand_raised = (left_wrist.y < left_shoulder.y and 
#                         left_wrist.y < nose.y + 0.1 and
#                         left_wrist.visibility > 0.5)
    
#     right_hand_raised = (right_wrist.y < right_shoulder.y and 
#                          right_wrist.y < nose.y + 0.1 and
#                          right_wrist.visibility > 0.5)
    
#     return left_hand_raised or right_hand_raised

# def is_focused(landmarks):
#     """Determine if student is focused based on head orientation and posture"""
#     # Get key landmarks
#     nose = landmarks[mp_pose.PoseLandmark.NOSE.value]
#     left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
#     right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
#     left_ear = landmarks[mp_pose.PoseLandmark.LEFT_EAR.value]
#     right_ear = landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value]
#     left_eye = landmarks[mp_pose.PoseLandmark.LEFT_EYE.value]
#     right_eye = landmarks[mp_pose.PoseLandmark.RIGHT_EYE.value]
    
#     # Calculate head tilt (vertical alignment)
#     head_center_y = nose.y
#     shoulder_center_y = (left_shoulder.y + right_shoulder.y) / 2
    
#     # Check if head is relatively upright (not looking down too much)
#     head_distance = head_center_y - shoulder_center_y
    
#     # Calculate horizontal head position (side tilt/turn)
#     ear_diff = abs(left_ear.x - right_ear.x)
    
#     # Check eye-to-shoulder alignment (looking down detection)
#     eye_center_y = (left_eye.y + right_eye.y) / 2
#     eye_to_shoulder = eye_center_y - shoulder_center_y
    
#     # Student is focused if:
#     # 1. Head is upright (not looking down excessively)
#     # 2. Head is relatively straight (not turned too far to side)
#     # 3. Eyes are not too close to shoulders (not slouching/looking down)
#     is_upright = head_distance < 0.15  # More sensitive threshold
#     is_straight = ear_diff > 0.04  # More sensitive to side turns
#     eyes_forward = eye_to_shoulder < 0.12  # Eyes not looking down
    
#     return is_upright and is_straight and eyes_forward

# def get_bounding_box(landmarks, image_width, image_height):
#     """Calculate bounding box around the person"""
#     x_coords = []
#     y_coords = []
    
#     for landmark in landmarks:
#         if landmark.visibility > 0.5:
#             x_coords.append(int(landmark.x * image_width))
#             y_coords.append(int(landmark.y * image_height))
    
#     if x_coords and y_coords:
#         x_min = max(0, min(x_coords) - 20)
#         x_max = min(image_width, max(x_coords) + 20)
#         y_min = max(0, min(y_coords) - 20)
#         y_max = min(image_height, max(y_coords) + 20)
#         return (x_min, y_min, x_max, y_max)
    
#     return None

# def main():
#     # Open webcam
#     cap = cv2.VideoCapture(0)
    
#     print("Student Monitor Started")
#     print("Press 'q' to quit")
    
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             print("Failed to grab frame")
#             break
        
#         # Flip frame horizontally for mirror view
#         frame = cv2.flip(frame, 1)
#         image_height, image_width, _ = frame.shape
        
#         # Convert to RGB for MediaPipe
#         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
#         # Process the frame
#         results = pose.process(rgb_frame)
        
#         # Count students detected
#         student_count = 0
        
#         if results.pose_landmarks:
#             student_count = 1  # One person detected per pose result
#             landmarks = results.pose_landmarks.landmark
            
#             # Get bounding box
#             bbox = get_bounding_box(landmarks, image_width, image_height)
            
#             if bbox:
#                 x_min, y_min, x_max, y_max = bbox
                
#                 # Determine focus status
#                 focused = is_focused(landmarks)
#                 color = (0, 255, 0) if focused else (0, 165, 255)  # Green or Orange
#                 status = "Focused" if focused else "Distracted"
                
#                 # Draw rectangle around person
#                 cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 3)
                
#                 # Draw status label
#                 label_y = y_min - 10 if y_min > 30 else y_max + 25
#                 cv2.putText(frame, status, (x_min, label_y), 
#                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
#                 # Check for raised hand
#                 if is_hand_raised(landmarks, image_width, image_height):
#                     # Draw "Raised Hand" text above the rectangle
#                     hand_label_y = y_min - 40 if y_min > 60 else y_max + 55
#                     cv2.putText(frame, "RAISED HAND", (x_min, hand_label_y), 
#                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    
#                     # Draw a hand icon/indicator
#                     cv2.circle(frame, (x_min - 15, hand_label_y - 5), 10, (0, 0, 255), -1)
        
#         # Display student count at top left
#         cv2.putText(frame, f"Students: {student_count}", (10, 30), 
#                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
#         # Display the frame
#         cv2.imshow('Student Focus Monitor', frame)
        
#         # Exit on 'q' press
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
    
#     # Cleanup
#     cap.release()
#     cv2.destroyAllWindows()
#     pose.close()

# if __name__ == "__main__":
#     main()



