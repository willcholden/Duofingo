# create dataset
import time

category = input("Enter category to create dataset: ")
start_time = time.time()

import mediapipe as mp
import cv2
import os
import matplotlib.pyplot as plt
import pickle 
from google.protobuf.json_format import MessageToDict



mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)


DATA_DIR = './data/' + category 

data = []
labels = []

zeros = [0] * 42

for item in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, item)):

        data_aux = []
        img = cv2.imread(os.path.join(DATA_DIR, item, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
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
            labels.append(item)

data_file = 'pickle_jar/' + category + '.pickle'
f=open(data_file, 'wb')
pickle.dump({'data': data, 'labels': labels}, f)
f.close()

print("--- %s seconds ---" %(time.time() - start_time))
