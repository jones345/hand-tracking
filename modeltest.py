import cv2
import mediapipe as mp
import time
import handtrackingmodel as model


ptime = 0
ctime = 0
cap = cv2.VideoCapture(0)
detector = model.HandDectotor()
while True:
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float `width`
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    success, image = cap.read()
    image = detector.findHands(image)
    lmlist = detector.findPosition(image, draw=False)
    if len(lmlist) != 0:
        print(lmlist[4])

    ctime = time.time()
    fps = 1 / (ctime - ptime)
    ptime = ctime
    cv2.putText(image, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 255), 3)
    cv2.imshow('image', image)
    cv2.waitKey(1)