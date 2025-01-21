import cv2
import mediapipe as mp
import numpy as np

# MediaPipe 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# 카메라 열기
cap = cv2.VideoCapture(0)

# 카운터 초기화
left_count = 0
right_count = 0
left_arm_up = False    # 왼팔이 위에 있는지 상태 체크용
right_arm_up = False   # 오른팔이 위에 있는지 상태 체크용

while cap.isOpened():
    # 카메라 프레임 읽기
    success, img = cap.read()
    if not success:
        break
        
    # 이미지를 RGB로 변환
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 포즈 감지
    results = pose.process(img_rgb)
    
    if results.pose_landmarks:
        # 포즈 그리기
        mp_draw.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        # 랜드마크에서 필요한 좌표 추출
        landmarks = results.pose_landmarks.landmark
        
        # 왼쪽 어깨와 손목의 y좌표
        left_shoulder_y = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y
        left_wrist_y = landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y
        
        # 오른쪽 어깨와 손목의 y좌표
        right_shoulder_y = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y
        right_wrist_y = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y
        
        # 손목이 어깨보다 얼마나 위에 있는지 계산
        left_diff = left_shoulder_y - left_wrist_y
        right_diff = right_shoulder_y - right_wrist_y
        
        # 왼팔 카운트
        if left_diff > 0.2 and not left_arm_up:  # 손목이 어깨보다 20% 이상 위에 있을 때
            left_count += 1
            left_arm_up = True
        elif left_diff < 0.15:  # 팔을 충분히 내렸을 때
            left_arm_up = False
            
        # 오른팔 카운트
        if right_diff > 0.2 and not right_arm_up:  # 손목이 어깨보다 20% 이상 위에 있을 때
            right_count += 1
            right_arm_up = True
        elif right_diff < 0.15:  # 팔을 충분히 내렸을 때
            right_arm_up = False
        
        # 화면에 카운트 표시
        cv2.putText(img, f"Left: {left_count}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(img, f"Right: {right_count}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    
    # 화면 표시
    cv2.imshow('Arm Counter', img)
    
    # q 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 종료
cap.release()
cv2.destroyAllWindows()

print(f"왼팔 든 횟수: {left_count}")
print(f"오른팔 든 횟수: {right_count}")