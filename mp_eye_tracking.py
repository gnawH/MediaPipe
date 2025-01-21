import cv2
import mediapipe as mp
import numpy as np

# MediaPipe Face Mesh 초기화
mp_face_mesh =  mp.solutions.face_mesh
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

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("카메라를 찾을 수 없습니다.")
        continue

    # BGR에서 RGB로 변환
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Face Mesh 감지 실행
    results = face_mesh.process(image)

    # RGB에서 BGR로 변환
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # 결과 데이터 출력
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # 왼쪽 눈 데이터
            print("\n=== LEFT EYE ===")
            for name, idx in LEFT_EYE_POINTS.items():
                landmark = face_landmarks.landmark[idx]
                h, w, _ = image.shape
                x, y = int(landmark.x * w), int(landmark.y * h)
                cv2.circle(image, (x, y), 2, (0, 0, 255), -1)
                print(f"{name}: ({x}, {y}), z: {landmark.z: .3f}")

            # 오른쪽 눈 데이터
            print("\n=== RIGHT EYE ===")
            for name, idx in RIGHT_EYE_POINTS.items():
                landmark = face_landmarks.landmark[idx]
                x, y = int(landmark.x * w), int(landmark.y * h)
                cv2.circle(image, (x, y), 2, (0, 0, 255), -1)
                print(f"{name}: ({x}, {y}), z: {landmark.z: .3f}")

        # 결과 화면 출력
        cv2.imshow("MediaPipe Face Mesh", image)

        # 종료 옵션
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

# 리소스 해제
cap.release()
cv2.destroyAllWindows()
face_mesh.close()