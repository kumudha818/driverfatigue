from tensorflow.keras.models import load_model
import cv2
import dlib
from scipy.spatial import distance
from ultralytics import YOLO
import winsound
import threading
import time
import datetime
import os


# Load models


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

yolo_model = YOLO("yolov8n.pt")
eye_model = load_model("models/eye_model.h5")


# Alarm function


def play_warning():
    winsound.Beep(800, 400)   # slow beep
    time.sleep(0.2)
    winsound.Beep(800, 400)


# EAR calculation


def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)


# MAR calculation


def mouth_aspect_ratio(mouth):
    A = distance.euclidean(mouth[13], mouth[19])
    B = distance.euclidean(mouth[14], mouth[18])
    C = distance.euclidean(mouth[15], mouth[17])
    D = distance.euclidean(mouth[12], mouth[16])
    return (A + B + C) / (2.0 * D)


# Thresholds


EAR_THRESHOLD = 0.22
MAR_THRESHOLD = 0.6

EYE_FRAMES = 12
YAWN_FRAMES = 10

EYE_COUNTER = 0
YAWN_COUNTER = 0



# Start camera


cap = cv2.VideoCapture(0)

last_beep_time = 0
frame_count = 0
cnn_eye_closed = False

print("Driver Fatigue Detection System Started")

log_file = open("fatigue_log.txt", "a")

last_log_time = 0
last_yawn_time = -1000
last_eye_close_time = -1000

while True:
    frame_count += 1

    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    eye_state = "Open"
    yawn_state = "No Yawn"
    phone_state = "Safe"
    head_state = "Forward"

    fatigue_score = 0

    
    # Face + Landmark detection
    

    for face in faces:
        face_gray = gray[face.top():face.bottom(), face.left():face.right()]

        if face_gray.size == 0:
           continue

        landmarks = predictor(gray, face)

        left_eye = []
        right_eye = []
        mouth = []

        for i in range(36,42):
            left_eye.append((landmarks.part(i).x, landmarks.part(i).y))

        for i in range(42,48):
            right_eye.append((landmarks.part(i).x, landmarks.part(i).y))

        for i in range(48,68):
            mouth.append((landmarks.part(i).x, landmarks.part(i).y))

        
        # EAR + MAR
        

        ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2
        mar = mouth_aspect_ratio(mouth)

        eye_img = cv2.resize(face_gray, (64,64))
        eye_img = eye_img / 255.0
        eye_img = eye_img.reshape(1,64,64,1)

        if frame_count % 3 == 0:
            eye_pred = eye_model.predict(eye_img, verbose=0)
            cnn_eye_closed = eye_pred[0][0] > 0.5

        # Eye detection

        if ear < EAR_THRESHOLD and cnn_eye_closed :
            EYE_COUNTER += 1
        else:
            EYE_COUNTER = 0

        if EYE_COUNTER > EYE_FRAMES:
            eye_state = "Closed"
            last_eye_close_time = time.time()

        # Yawn detection

        if mar > 0.7:
           YAWN_COUNTER += 2   # strong yawn
        elif mar > MAR_THRESHOLD:
           YAWN_COUNTER += 1   # normal yawn
        else:
           YAWN_COUNTER -= 1   # decrease slowly

        # Keep counter in range
        YAWN_COUNTER = max(0, min(YAWN_COUNTER, 20))

        # FINAL DECISION
        if YAWN_COUNTER >= 8:
           yawn_state = "Yawning"
           last_yawn_time = time.time()
        else:
           yawn_state = "No Yawn"

    
    # Phone detection (YOLO)
    

    results = yolo_model(frame)

    for r in results:
        for box in r.boxes:

            cls = int(box.cls[0])
            label = yolo_model.names[cls]

            if label == "cell phone":

                phone_state = "Phone Detected"

                x1,y1,x2,y2 = map(int,box.xyxy[0])
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),2)

    
    # Fatigue score
    

    if eye_state == "Closed":
        fatigue_score += 40

    if yawn_state == "Yawning":
        fatigue_score += 30

    if phone_state == "Phone Detected":
        fatigue_score += 30

    
    # Alert system
    
    current_time = time.time()

    if fatigue_score >= 60:
        alert = "CRITICAL ALERT"
        color = (0,0,255)

    elif fatigue_score >= 30:

        alert = "WARNING"
        color = (0,165,255)
     
    else:

        alert = "NORMAL"
        color = (0,255,0)

    if (current_time - last_yawn_time < 60) and \
       (current_time - last_eye_close_time < 60) and \
       (eye_state == "Closed" or yawn_state == "Yawning"):

        alert = "CRITICAL ALERT"
        color = (0,0,255)
        

        current_time = time.time()
        last_beep_time = current_time


     
     # LOG ONLY WARNING & CRITICAL
     

    if fatigue_score >= 30:

       current_time_str = datetime.datetime.now().strftime("%H:%M:%S")

       log_text = f"Time: {current_time_str} | Score: {fatigue_score} | {alert}\n"

       print("LOG:", log_text)   # debug (you will see in terminal)

       log_file.write(log_text)
       log_file.flush()

    
    # Display
    

    cv2.putText(frame,f"Eye: {eye_state}",(30,40),
                cv2.FONT_HERSHEY_SIMPLEX,0.8,color,2)

    cv2.putText(frame,f"Yawn: {yawn_state}",(30,80),
                cv2.FONT_HERSHEY_SIMPLEX,0.8,color,2)

    cv2.putText(frame,f"Phone: {phone_state}",(30,120),
                cv2.FONT_HERSHEY_SIMPLEX,0.8,color,2)

    cv2.putText(frame,f"Fatigue Score: {fatigue_score}",(30,200),
                cv2.FONT_HERSHEY_SIMPLEX,0.8,color,2)

    cv2.putText(frame,alert,(30,250),
                cv2.FONT_HERSHEY_SIMPLEX,1,color,3)

    cv2.imshow("Driver Fatigue Detection",frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
log_file.close()