import cv2
import mediapipe as mp
import numpy as np

# MediaPipe Face Mesh 초기화
mp_face_mesh =  mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces = 1,
    min_detection_confidence = 0.5,
    min_tracking_confidence = 0.5
)

# 웹캠 초기화
# VScode Superuser로 실행
cap = cv2.VideoCapture(0)

# 눈 랜드마크 포인트 정의
LEFT_EYE_POINTS = {
    '눈꺼풀 위': 386,  # 왼쪽 눈 위쪽
    '눈꺼풀 아래': 374,  # 왼쪽 눈 아래
    '눈 안쪽': 362,  # 왼쪽 눈 안쪽 모서리
    '눈 바깥쪽': 263,  # 왼쪽 눈 바깥쪽 코너
}

RIGHT_EYE_POINTS = {
    '눈꺼풀 위': 159,  # 오른쪽 눈 위쪽
    '눈꺼풀 아래': 145,  # 오른쪽 눈 아래
    '눈 안쪽': 133,  # 오른쪽 눈 안쪽 모서리
    '눈 바깥쪽': 33,  # 오른쪽 눈 바깥쪽 코너
}

def calculate_eye_aspect_ratio(landmarks, eye_points):
    # 눈 수직 거리 계산
    upper = landmarks.landmark[eye_points['눈꺼풀 위']]
    lower = landmarks.landmark[eye_points['눈꺼풀 아래']]
    height = abs(upper.y - lower.y)

    # 눈 수평 거리 계산
    inner = landmarks.landmark[eye_points['눈 안쪽']]
    outer = landmarks.landmark[eye_points['눈 바깥쪽']]
    width = abs(outer.x - inner.x)

    # 눈 종횡비(EAR) 계산
    ear = height / width
    return ear

# EAR 임계값 설정
EAR_THRESHOLD = 0.4

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("카메라를 찾을 수 없습니다.")
        continue

    # 이미지 좌우 반전(거울 모드)
    image = cv2.flip(image, 1)

    # BGR을 RGB로 변환
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Face Mesh 감지 실행
    results = face_mesh.process(rgb_image)

    # 결과 데이터 출력 및 시각화
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # 전체 Face Mesh 그리기 
            mp_drawing.draw_landmarks(
                image = image,
                landmark_list = face_landmarks,
                connections = mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec = mp_drawing.DrawingSpec(color = (0, 255, 0), thickness = 1, circle_radius = 1),
                connection_drawing_spec = mp_drawing.DrawingSpec(color = (0, 255, 0), thickness = 1)
            )

            # 눈 포인트 표시 및 EAR 계산
            h, w, _ = image.shape

            # 왼쪽 눈
            left_ear = calculate_eye_aspect_ratio(face_landmarks, LEFT_EYE_POINTS)
            for name, idx in LEFT_EYE_POINTS.items():
                pos = face_landmarks.landmark[idx]
                x, y = int(pos.x * w), int(pos.y * h)
                cv2.circle(image, (x, y), 2, (0, 0, 255), -1)

            # 오른쪽 눈
            right_ear = calculate_eye_aspect_ratio(face_landmarks, RIGHT_EYE_POINTS)
            for name, idx in RIGHT_EYE_POINTS.items():
                pos = face_landmarks.landmark[idx]
                x, y = int(pos.x * w), int(pos.y * h)
                cv2.circle(image, (x, y), 2, (0, 0, 255), -1)

            # 평균 EAR 계산
            avg_ear = (left_ear + right_ear) / 2

            # EAR 값 화면에 표시
            cv2.putText(image, f"EAR: {avg_ear: .2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # 눈 감김 상태 표시
            if avg_ear < EAR_THRESHOLD:
                cv2.putText(image, "Eyes Closed!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                cv2.putText(image, "Eyes Open", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            print(f"\r왼쪽 EAR: {left_ear: .3f}, 오른쪽 EAR: {right_ear: .3f}, 평균 EAR: {avg_ear: .3f}", end='')

        # 화면 출력
        cv2.imshow("Eye Tracking", image)

        # 종료 옵션
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
    
cap.release()
cv2.destroyAllWindows()
face_mesh.close()