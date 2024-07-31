import cv2 as cv
import time
import utils
import mediapipe as mp
import numpy as np
import warnings
warnings.filterwarnings("ignore", message="SymbolDatabase.GetPrototype() is deprecated.")

mp_face_detection = mp.solutions.face_detection

cap = cv.VideoCapture(0)
with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
    frame_counter = 0
    start_time = time.time()
    font  = cv.FONT_HERSHEY_PLAIN
    while True:
        frame_counter  +=1
        ret, frame = cap.read()
        if ret is False:
            break

        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = face_detection.process(rgb_frame)
        frame_width, frame_height, c = frame.shape
        if results.detections:
            for face in results.detections:
                face_react = np.multiply(
                    [face.location_data.relative_bounding_box.xmin,
                    face.location_data.relative_bounding_box.ymin,
                    face.location_data.relative_bounding_box.width,
                    face.location_data.relative_bounding_box.height],
                    [frame_width, frame_height, frame_width, frame_height]
                ).astype(int)
            utils.rect_corners(frame, face_react, utils.YELLOW,th=3)
            # print(face_reaction)
        fps = frame_counter/(time.time() - start_time)
        utils.text_with_background(frame, f"FPS: {fps:.2f}", (30, 30),font)
        

        cv.imshow('window', frame)
        

        key = cv.waitKey(1)
        if key == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()
