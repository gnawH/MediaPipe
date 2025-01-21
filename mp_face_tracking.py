import cv2
import mediapipe as mp
import time

# MediaPipe Face Mesh 초기화
mp_face_mesh =  mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces = 1,  # 감지할 최대 얼굴 수
    min_detection_confidence = 0.5,
    min_tracking_confidence = 0.5
)

# MediaPipe Drawing 유틸리티 초기화
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness = 1, circle_radius = 1)

# 웹캠 초기화
cap = cv2.VideoCapture(0)
while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("카메라를 찾을 수 없습니다.")
        continue

    # BGR에서 RGB로 변환
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Face Mesh 감지 실행
    results = face_mesh.process(image)

    # RGB에서 BGR로 다시 변환
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Mesh 그리기
            mp_drawing.draw_landmarks(
                image = image,
                landmark_list = face_landmarks,
                connections = mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec = drawing_spec,
                connection_drawing_spec = drawing_spec
            )

            # 윤곽선 그리기
            mp_drawing.draw_landmarks(
                image = image,
                landmark_list = face_landmarks,
                connections = mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec = drawing_spec,
                connection_drawing_spec = drawing_spec
            )

    # 결과 화면 출력
    cv2.imshow("MediaPipe Face Mesh", image)

    # 종료 옵션
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# 리소스 해제
cap.release()
cv2.destroyAllWindows()
face_mesh.close()