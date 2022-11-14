import cv2
import mediapipe as mp
import time

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,1024)
sTime =time.time()
with mp_hands.Hands(
        min_detection_confidence=0.5,min_tracking_confidence=0.5) as hands:
    while cap.isOpened():

        success, image = cap.read()
        # Flip the image horizontally for a later selfie-view display
        # Convert the BGR image to RGB.
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

        # To improve performance, optionally mark the image as not writeable
        image.flags.writeable = False

        # Process the image and find hands
        results = hands.process(image)

        image.flags.writeable = True

        # Draw the hand annotations on the image.
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for id, lm in enumerate(hand_landmarks.landmark):
                    print("Landmark", id)
                    print(lm)
                    print("Time stamp", time.strftime('%H:%M:%S'))
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                          mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2),
                                          )

        eTime = time.time()
        fps = 1 / (eTime - sTime)
        sTime=eTime
        cv2.putText(image, f'FPS:{int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
        cv2.imshow('Hands Mediapipe', image)

        if cv2.waitKey(5) & 0xFF == ord('q'):
          break

cap.release()