import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils
while True:
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float `width`
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    success, image = cap.read()
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    results = hands.process(imageRGB)
    print(results.multi_hand_landmarks)
    if results.multi_hand_landmarks:
        for handlm in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(image, handlm, mpHands.HAND_CONNECTIONS)

    cv2.imshow('image', image)
    cv2.waitKey(1)
