import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Initialize YOLO for person detection
model = YOLO('yolov8n.pt')  # Using nano model for speed, can use yolov8s.pt for better accuracy

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
    avg_eye_y = (l_eye.y + r_eye.y) / 2
    avg_ear_y = (l_ear.y + r_ear.y) / 2
    
    is_not_looking_down = avg_eye_y < (avg_ear_y - 0.01)

    # 2. DETECT SIDE TURN (Eye Symmetry)
    dist_l = abs(nose.x - l_eye.x)
    dist_r = abs(nose.x - r_eye.x)
    
    eye_ratio = dist_l / (dist_r + 1e-6)
    is_facing_forward = 0.25 < eye_ratio < 4.0

    # 3. DETECT SLUMPING (Nose vs Shoulder)
    avg_shoulder_y = (l_shoulder.y + r_shoulder.y) / 2
    is_not_slumping = nose.y < (avg_shoulder_y - 0.05)

    return is_not_looking_down and is_facing_forward and is_not_slumping

def analyze_person(person_crop, bbox):
    """Analyze a single person's pose and return status"""
    # Initialize pose estimator for this person
    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=0  # Use lighter model for speed with multiple people
    ) as pose:
        
        # Convert to RGB
        rgb_crop = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_crop)
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # Get crop dimensions
            crop_height, crop_width = person_crop.shape[:2]
            
            # Check focus status
            focused = is_focused(landmarks)
            
            # Check hand raised
            hand_raised = is_hand_raised(landmarks, crop_width, crop_height)
            
            return {
                'focused': focused,
                'hand_raised': hand_raised,
                'landmarks': landmarks
            }
    
    return None

def main():
    cap = cv2.VideoCapture(0)
    print("Multi-Student Monitor Started - Press 'q' to quit")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        image_height, image_width, _ = frame.shape
        
        # Detect all people in frame using YOLO
        results = model(frame, classes=[0], verbose=False)  # class 0 is 'person'
        
        student_count = 0
        focused_count = 0
        distracted_count = 0
        hands_raised = 0
        
        # Process each detected person
        for result in results:
            boxes = result.boxes
            
            for box in boxes:
                # Get bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                
                # Only process high-confidence detections
                if confidence < 0.5:
                    continue
                
                student_count += 1
                
                # Add padding to bounding box
                padding = 20
                x1 = max(0, x1 - padding)
                y1 = max(0, y1 - padding)
                x2 = min(image_width, x2 + padding)
                y2 = min(image_height, y2 + padding)
                
                # Extract person crop
                person_crop = frame[y1:y2, x1:x2]
                
                if person_crop.size == 0:
                    continue
                
                # Analyze this person's pose
                analysis = analyze_person(person_crop, (x1, y1, x2, y2))
                
                if analysis:
                    focused = analysis['focused']
                    hand_raised = analysis['hand_raised']
                    
                    # Update counts
                    if focused:
                        focused_count += 1
                    else:
                        distracted_count += 1
                    
                    if hand_raised:
                        hands_raised += 1
                    
                    # Draw bounding box with appropriate color
                    color = (0, 255, 0) if focused else (0, 165, 255)
                    status = "Focused" if focused else "DISTRACTED"
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                    
                    # Draw status label
                    label_y = y1 - 10 if y1 > 30 else y2 + 25
                    cv2.putText(frame, status, (x1, label_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    
                    # Draw hand raised indicator
                    if hand_raised:
                        hand_label_y = y1 - 40 if y1 > 60 else y2 + 55
                        cv2.putText(frame, "RAISED HAND", (x1, hand_label_y), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                        cv2.circle(frame, (x1 - 15, hand_label_y - 5), 10, (0, 0, 255), -1)
                else:
                    # If pose analysis failed, just draw a neutral box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (128, 128, 128), 2)
                    cv2.putText(frame, "Detecting...", (x1, y1 - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 2)
        
        # Draw statistics panel
        cv2.rectangle(frame, (10, 10), (350, 130), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (350, 130), (255, 255, 255), 2)
        
        cv2.putText(frame, f"Total Students: {student_count}", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Focused: {focused_count}", (20, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Distracted: {distracted_count}", (20, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        
        if hands_raised > 0:
            cv2.putText(frame, f"Hands Raised: {hands_raised}", (200, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.imshow('Multi-Student Focus Monitor', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


