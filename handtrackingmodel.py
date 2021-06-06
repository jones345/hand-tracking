import cv2
import mediapipe as mp
import time


class HandDectotor():
    def __init__(self, model=False, maxhands=2, detectionCon=0.5, trackCon=0.5):
        self.model = model
        self.maxHands = maxhands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.model, self.maxHands, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, image, draw=True):
        imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        self.results = self.hands.process(imageRGB)
        # print(results.multi_hand_landmarks)
        if self.results.multi_hand_landmarks:
            for handlm in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(image, handlm, self.mpHands.HAND_CONNECTIONS)

        return image

    def findPosition(self, image, HandNo=0, draw=True):
        lmlist = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[HandNo]
            for id, lm in enumerate(myHand.landmark):
                # print(id, lm)
                h, w, c = image.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                # print(id, cx, cy)
                lmlist.append([id, cx, cy])
                if draw:
                    cv2.circle(image, (cx, cy), 7, (255, 0, 0), cv2.FILLED)

        return lmlist


def main():
    ptime = 0
    ctime = 0
    cap = cv2.VideoCapture(0)
    detector = HandDectotor()
    while True:
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float `width`
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        success, image = cap.read()
        image = detector.findHands(image)
        lmlist = detector.findPosition(image)
        if len(lmlist) != 0:
            print(lmlist[4])

        ctime = time.time()
        fps = 1 / (ctime - ptime)
        ptime = ctime
        cv2.putText(image, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 255), 3)
        cv2.imshow('image', image)
        cv2.waitKey(1)


if __name__ == '__main__':
    main()
