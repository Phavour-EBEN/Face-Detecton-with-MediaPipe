import cv2 as cv
import time
import utils
import mediapipe as mp

font  = cv.FONT_HERSHEY_PLAIN
cap = cv.VideoCapture(0)
frame_counter = 0
start_time = time.time()

while True:
    frame_counter  +=1
    ret, frame = cap.read()
    if ret is False:
        break
    fps = frame_counter/(time.time() - start_time)
    utils.text_with_background(frame, f"FPS: {fps:.2f}", (30, 30),font)
    

    cv.imshow('window', frame)
    

    key = cv.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
