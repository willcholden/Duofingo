# create dataset USING FRAMES
category = input("Enter category to create dataset: ")

import mediapipe as mp
import cv2
import pickle 
from google.protobuf.json_format import MessageToDict
import os
import time 

cap = cv2.VideoCapture(1)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)



data_file = './pickle_jar/' + category + '.pickle'

if os.path.exists(data_file):
    data_dict = pickle.load(open(data_file, 'rb'))
else:
    data_dict = {}

zeros = [0] * 42
while True:
    item = input("Enter item OR Enter 'quit' to quit: ")
    if item == "quit":
        break
    if item == "delete":
        delete_item = input("Enter item to delete: ")
        del data_dict[delete_item]
        break 
    counter = 0
    data = []

    while True: 
        str1 = "Press 'q' when ready to record: " + item
        ret, frame = cap.read()

        cv2.rectangle(frame, 
            (200, 240),    # (x1, y1) Top left corner
            (440, 480),   # (x2, y2) Bottom right corner
            (0,0,250), 
            1)
                
        cv2.rectangle(frame, 
            (840, 240), 
            (1080, 480), 
            (0,0,250), 
            1)

        cv2.putText(frame, 
                    str1, 
                    (100, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1.3, 
                    (0, 255, 0), 
                    3, 
                    cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('q'): 
            break

    time.sleep(3)

    while counter < 100:

        ret, frame = cap.read()
        height, width = frame.shape[:2]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        cv2.rectangle(frame, 
            (200, 240),    # (x1, y1) Top left corner
            (440, 480),   # (x2, y2) Bottom right corner
            (0,0,250), 
            4)
            
        cv2.rectangle(frame, 
            (840, 240), 
            (1080, 480), 
            (0,0,250), 
            4)

        data_aux = []

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame,
                                      hand_landmarks,
                                      mp_hands.HAND_CONNECTIONS,
                                      mp_drawing_styles.get_default_hand_landmarks_style(),
                                      mp_drawing_styles.get_default_hand_connections_style())
                
                for i in range(len(hand_landmarks.landmark)):
                    x=hand_landmarks.landmark[i].x
                    y=hand_landmarks.landmark[i].y
                    data_aux.append(x)
                    data_aux.append(y)


            handedness = results.multi_handedness
            if len(handedness) == 1:
                handedness_dict = MessageToDict(handedness[0])
                which_hand = handedness_dict['classification'][0]['label']

                if which_hand == "Left": 
                    data_aux = data_aux + zeros
                else:
                    data_aux = zeros + data_aux

            data.append(data_aux)
            counter += 1

        cv2.imshow('frame', frame)
        cv2.waitKey(50)

    data_dict[item] = data

# -------------------------------------------------------------------------------------------------
#                  { item :       data }
#   animals_dict = {'deer':    [ [0.12, 0.72,...], [0.12, 0.72,...],...,[0.12, 0.72,...] ],
#                   'coyote':  [ [0.43, 0.95,...], [0.41, 0.96,...],...,[0.45, 0.94,...] ],
#                   ...
#                   'opossom': [ [0.05, 0.55,...], [0.07, 0.54,...],...,[0.06, 0.51,...] ],
#                   }
#

f=open(data_file, 'wb')
pickle.dump(data_dict, f)
f.close()
