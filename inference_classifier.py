# test classifier
import warnings
warnings.filterwarnings("ignore")

import cv2
import mediapipe as mp
import pickle
import numpy as np
from google.protobuf.json_format import MessageToDict

category = input("Enter category to predict: ")

model_name = 'pickle_jar/' +  category + '_model.p'
model_dict = pickle.load(open(model_name, 'rb'))
model = model_dict['model']

# _________________________________________________________________________________________________
# CATEGORY LISTS
# -------------------------------------------------------------------------------------------------
cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

zeros = [0] * 42
prediction_queue = ['NA'] * 10


while True:

    data_aux = []
    x_ = []
    y_ = []

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
    
    H, W, _ = frame.shape 

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame,
                                      hand_landmarks,
                                      mp_hands.HAND_CONNECTIONS,
                                      mp_drawing_styles.get_default_hand_landmarks_style(),
                                      mp_drawing_styles.get_default_hand_connections_style())
            
            
        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x=hand_landmarks.landmark[i].x
                y=hand_landmarks.landmark[i].y
                data_aux.append(x)
                data_aux.append(y)
                x_.append(x)
                y_.append(y)

        handedness = results.multi_handedness
        if len(handedness) == 1:
            handedness_dict = MessageToDict(handedness[0])
            which_hand = handedness_dict['classification'][0]['label']

            if which_hand == "Left": 
                data_aux = data_aux + zeros
            else:
                data_aux = zeros + data_aux 

        if len(data_aux) == 84:
            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10
            x2 = int(max(x_) * W) - 10
            y2 = int(max(y_) * H) - 10
            prediction = model.predict([np.asarray(data_aux)])
            
            predicted_character = prediction[0]
            if max(model.predict_proba([np.asarray(data_aux)])[0]) < 0.25:
                predicted_character = ""


            prediction_queue.insert(0, predicted_character)
            prediction_queue.pop()
            # print(prediction_queue)

            # _____________________________________________________________________________________
            # GESTURES WITH MOVEMENT
            # -------------------------------------------------------------------------------------

            # Letters
            if category == 'letters':
                if prediction_queue[0] == 'z2':
                    if prediction_queue[9] == 'z1':
                        predicted_character = 'z'

                if prediction_queue[0] == 'j2':
                    if prediction_queue[9] == 'i':
                        predicted_character = 'j'

            # Animals
            if category == 'animals':
                if prediction_queue[0] == 'fish2':
                    if prediction_queue[9] == 'fish1':
                        predicted_character = 'fish'

                if prediction_queue[0] == 'frog2':
                    if prediction_queue[9] == 'frog1':
                        predicted_character = 'frog'

                if prediction_queue[0] == 'pig2':
                    if prediction_queue[9] == 'pig1':
                        predicted_character = 'pig'

                if prediction_queue[0] == 'mouse2':
                    if prediction_queue[9] == 'mouse1':
                        predicted_character = 'mouse'

                if prediction_queue[0] == 'raccoon2':
                    if prediction_queue[9] == 'raccoon1':
                        predicted_character = 'raccoon'

                if prediction_queue[0] == 'snake2':
                    if prediction_queue[9] == 'snake1':
                        predicted_character = 'snake'

            # _____________________________________________________________________________________
            # 
            # -------------------------------------------------------------------------------------

            
            cv2.putText(frame, 
                    predicted_character,
                    (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1.3, 
                    (230, 230, 250), 
                    3, 
                    cv2.LINE_AA)


    cv2.imshow('frame', frame)
    cv2.waitKey(100)





cap.release()
cv2.destroyAllWindows()
