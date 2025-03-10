import mediapipe as mp
import cv2

class HandDetector:
    def __init__(self,
                 mode=False,
                 max_hands=2,
                 detection_conf=0.5,
                 track_conf=0.5):

        self.lmList = None
        self.results = None
        self.hand_params = {
            "static_image_mode":mode,
            "max_num_hands":max_hands,
            "min_detection_confidence": detection_conf,
            "min_tracking_confidence": track_conf
        }

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(**self.hand_params)
        self.mpDraw = mp.solutions.drawing_utils

        self.tipIds = [4, 8, 12, 16, 20]

    def find_hands(self, img, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img,
                                               handLms,
                                               self.mpHands.HAND_CONNECTIONS,
                                               self.mpDraw.DrawingSpec((255,0,0), 2, 3),
                                               self.mpDraw.DrawingSpec((0,255,0), 2))
        return img

    def find_positions(self, img, hand_no=0, draw=True):
        self.lmList = []

        if self.results.multi_hand_landmarks:
            my_hand = self.results.multi_hand_landmarks[hand_no]
            for idi, lm in enumerate(my_hand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([idi, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 10, (255,0,255), -1)
        return self.lmList

    def fingers_up(self):
        fingers = []

        # if self.lmList[17][1] < self.lmList[2][1]:
        # Right Thumb - This is for the cv2.flip(img, 1) image
        if self.lmList[self.tipIds[0]][1] < self.lmList[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        # else:
        # Left Thumb
        # if self.lmList[tipIds[0]][1] < self.lmList[tipIds[0] - 1][1]:
        #     fingers.append(1)
        # else:
        #     fingers.append(0)

        # Four Fingers Right
        for i in range(1, len(self.tipIds)):
            if self.lmList[self.tipIds[i]][2] < self.lmList[self.tipIds[i] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers
