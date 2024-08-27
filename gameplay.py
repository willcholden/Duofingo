# GAMEPLAY
import warnings
warnings.filterwarnings("ignore")

import cv2
import mediapipe as mp
import pickle
import numpy as np
from google.protobuf.json_format import MessageToDict
import time
import random
import sys

#category = input("Enter category to predict: ")


# _________________________________________________________________________________________________
# CATEGORY LISTS
# _________________________________________________________________________________________________


item_lists = {'letters': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'],
              'animals': ['bird', 'coyote', 'deer', 'fish', 'frog', 'mouse', 'opossum', 'rabbit', 'raccoon', 'snake', 'squirrel', 'turtle', 'pig'],
              'foods': ['apple', 'egg', 'hamburger', 'water', 'milk', 'peach', 'cheese', 'bread', 'cake', 'orange', 'pear', 'butter', 'corn', 'potato'],
              'basics': ['hello', 'me', 'father', 'mother', 'yes', 'no', 'help', 'please', 'thank you', 'want', 'what', 'repeat', 'more', 'fine', 'learn', 'sign', 'finish'],
              'verbs': ['eat', 'drink', 'walk', 'run', 'sit', 'stand', 'sleep', 'stop', 'wash', 'drop', 'drive', 'jump', 'open', 'throw']}


cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

zeros = [0] * 42
prediction_queue = ['NA'] * 5
predicted_character = ""

# _________________________________________________________________________________________________
# MAIN MENU
# _________________________________________________________________________________________________

def mouseClick(event, x, y, flags, param):
    global category, image, window_name 
    if event == cv2.EVENT_LBUTTONDOWN:
        image2 = image.copy()

        # LETTERS
        if (x > 40) & (x < 240) & (y > 300) & (y < 550):
            category = 'letters'
            cv2.rectangle(image2, (40, 300), (240, 550), (189, 200, 135), 2)
            cv2.imshow(window_name, image2)

        # BASICS
        elif (x > 325) & (x < 525) & (y > 300) & (y < 550):
            category = 'basics'
            cv2.rectangle(image2, (325, 300), (525, 550), (189, 200, 135), 2)
            cv2.imshow(window_name, image2)

        # FOODS
        elif (x > 540) & (x < 740) & (y > 300) & (y < 550):
            category = 'foods'
            cv2.rectangle(image2, (540, 300), (740, 550), (189, 200, 135), 2)
            cv2.imshow(window_name, image2)

        # VERBS
        elif (x > 780) & (x < 980) & (y > 300) & (y < 550):
            category = 'verbs'
            cv2.rectangle(image2, (780, 300), (980, 550), (189, 200, 135), 2)
            cv2.imshow(window_name, image2)

        # ANIMALS
        elif (x > 1020) & (x < 1220) & (y > 300) & (y < 550):
            category = 'animals'
            cv2.rectangle(image2, (1020, 300),  (1220, 550),  (189, 200, 135), 2)
            cv2.imshow(window_name, image2)

    
while True:

    category = ''
    path = './logo.png'
    image = cv2.imread(path)
    
    window_name = 'Duofingo'
    cv2.namedWindow(window_name)
    cv2.resizeWindow(window_name, 1280, 720) 
    cv2.setMouseCallback(window_name, mouseClick)

    while True:
        cv2.imshow(window_name, image)
        if cv2.waitKey(0) and category != '':
            break

    cv2.destroyAllWindows()

    model_name = 'pickle_jar/' +  category + '_model.p'
    model_dict = pickle.load(open(model_name, 'rb'))
    model = model_dict['model']
    target = item_lists[category][random.randint(0, len(item_lists[category])-1)]

    total_time = 60
    start_time = time.time()
    curr_time = total_time
    score = 0

    while curr_time > 0:
        ret, frame = cap.read()

        data_aux = []
        x_ = []
        y_ = []


        ellapsed_time = time.time() - start_time
        curr_time = total_time - ellapsed_time
        


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
                if max(model.predict_proba([np.asarray(data_aux)])[0]) < 0.35:
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
                        if prediction_queue[4] == 'z1':
                            predicted_character = 'z'

                    if prediction_queue[0] == 'j2':
                        if prediction_queue[4] == 'i':
                            predicted_character = 'j'

                # Animals
                if category == 'animals':
                    if prediction_queue[0] == 'fish2':
                        if prediction_queue[4] == 'fish1':
                            predicted_character = 'fish'

                    if prediction_queue[0] == 'frog2':
                        if prediction_queue[4] == 'frog1':
                            predicted_character = 'frog'

                    if prediction_queue[0] == 'mouse2':
                        if prediction_queue[4] == 'mouse1':
                            predicted_character = 'mouse'

                    if prediction_queue[0] == 'raccoon2':
                        if prediction_queue[4] == 'raccoon1':
                            predicted_character = 'raccoon'

                    if prediction_queue[0] == 'snake2':
                        if prediction_queue[4] == 'snake1':
                            predicted_character = 'snake'

                    if prediction_queue[0] == 'pig2':
                        if prediction_queue[4] == 'pig1':
                            predicted_character = 'pig'

                # FOODS
                if category == 'foods':
                    if prediction_queue[0] == 'apple2':
                        if 'apple1' in prediction_queue:
                            predicted_character = 'apple'

                    if prediction_queue[0] == 'cake1':
                        if 'cookies1' in prediction_queue:
                            predicted_character = 'cookie'
                    
                    if prediction_queue[0] == 'egg2':
                        if 'egg1' in prediction_queue:
                            predicted_character = 'egg'

                    if prediction_queue[0] == 'hamburger2':
                        if 'hamburger1' in prediction_queue or 'cake1' in prediction_queue:
                            predicted_character = 'hamburger'

                    if prediction_queue[0] == 'peach2':
                        if 'peach1' in prediction_queue:
                            predicted_character = 'peach'

                    if prediction_queue[0] == 'bread2':
                        if 'bread1' in prediction_queue:
                            predicted_character = 'bread'

                    if prediction_queue[0] == 'cake2':
                        if 'cake1' in prediction_queue or 'hamburger1' in prediction_queue:
                            predicted_character = 'cake'

                    if prediction_queue[0] == 'pear2':
                        if 'pear1' in prediction_queue:
                            predicted_character = 'pear'

                    if prediction_queue[0] == 'corn2':
                        if 'corn1' in prediction_queue:
                            predicted_character = 'corn'

                # BASICS
                if category == 'basics':
                    if prediction_queue[0] == 'hello2':
                        if 'hello1' in prediction_queue:
                            predicted_character = 'hello'

                    if prediction_queue[0] == 'yes2':
                        if 'yes1' in prediction_queue:
                            predicted_character = 'yes'
                            
                    if prediction_queue[0] == 'no2':
                        if 'no1' in prediction_queue:
                            predicted_character = 'no'
                            
                    if prediction_queue[0] == 'please2':
                        if 'please1' in prediction_queue:
                            predicted_character = 'please'
                            
                    if prediction_queue[0] == 'thanks2':
                        if 'thanks1' in prediction_queue:
                            predicted_character = 'thank you'
                            
                    if prediction_queue[0] == 'no2':
                        if 'no1' in prediction_queue:
                            predicted_character = 'no'
                            
                    if prediction_queue[0] == 'want':
                        if 'what1' in prediction_queue:
                            predicted_character = 'what'
                            
                    if prediction_queue[0] == 'learn2':
                        if 'learn1' in prediction_queue:
                            predicted_character = 'learn'
                            
                    if prediction_queue[0] == 'sign2':
                        if 'sign1' in prediction_queue:
                            predicted_character = 'sign'
                            
                    if prediction_queue[0] == 'finish2':
                        if 'finish1' in prediction_queue:
                            predicted_character = 'finish'

                # VERBS
                if category == 'verbs':
                    if prediction_queue[0] == 'drink2':
                        if 'drink1' in prediction_queue:
                            predicted_character = 'drink'
                            
                    if prediction_queue[0] == 'walk2':
                        if 'walk1' in prediction_queue:
                            predicted_character = 'walk'
                            
                    if prediction_queue[0] == 'run2':
                        if 'run1' in prediction_queue:
                            predicted_character = 'run'
                            
                    if prediction_queue[0] == 'eat':
                        if 'sleep1' in prediction_queue:
                            predicted_character = 'sleep'
                            
                    if prediction_queue[0] == 'drop2':
                        if 'drop1' in prediction_queue:
                            predicted_character = 'drop'
                            
                    if prediction_queue[0] == 'drive2':
                        if 'drive1' in prediction_queue:
                            predicted_character = 'drive'
                            
                    if prediction_queue[0] == 'jump2':
                        if 'stand' in prediction_queue:
                            predicted_character = 'jump'
                            
                    if prediction_queue[0] == 'open2':
                        if 'open1' in prediction_queue:
                            predicted_character = 'open'
                            
                    if prediction_queue[0] == 'open1':
                        if 'open2' in prediction_queue:
                            predicted_character = 'close'
                            
                    if prediction_queue[0] == 'throw2':
                        if 'throw1' in prediction_queue:
                            predicted_character = 'throw'
                    

                # _____________________________________________________________________________________
                # 
                # -------------------------------------------------------------------------------------

                
                # cv2.putText(frame, 
                #         predicted_character,
                #         (x1, y1 - 10), 
                #         cv2.FONT_HERSHEY_SIMPLEX, 
                #         1.3, 
                #         (230, 230, 250), 
                #         3, 
                #         cv2.LINE_AA)
                
        if predicted_character == target:
            score += 1
            target = item_lists[category][random.randint(0, len(item_lists[category])-1)]  

        cv2.rectangle(frame, (0, 645), (1280, 720), (209, 222, 150), -1)
                
        cv2.putText(frame, 
                "target: " + target, 
                (50, 695), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1.0, 
                (170, 180, 120), 
                3, 
                cv2.LINE_AA)
        
        cv2.putText(frame, 
                "score: " + str(score), 
                (550, 695), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1.0, 
                (170, 180, 120), 
                3, 
                cv2.LINE_AA)
        
        cv2.putText(frame, 
        "time remaining: " + str(curr_time)[0:4],
        (900, 695), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        1.0, 
        (170, 180, 120), 
        3, 
        cv2.LINE_AA)


        cv2.imshow('frame', frame)
        cv2.waitKey(100)
    cv2.destroyAllWindows()



# # _________________________________________________________________________________________________
# # Exit Screen
# # _________________________________________________________________________________________________

    # HIGH SCORES
    f=open('./pickle_jar/highscores.pickle', 'rb')
    highscores_dict = pickle.load(f)
    f.close()
    
    if score <= highscores_dict[category]:
        exit_script1 = "Score: " + str(score)
        exit_script2 = "High score: " + str(highscores_dict[category])
    else:
        exit_script1 = "New high score: " + str(score) 
        exit_script2 = "Previous high score: " + str(highscores_dict[category])

        highscores_dict[category] = score
        f=open('./pickle_jar/highscores.pickle', 'wb')
        pickle.dump(highscores_dict, f)
        f.close()




    def exitOption(event, x, y, flags, param):
        global decision, image, window_name 
        if event == cv2.EVENT_LBUTTONDOWN:
            image2 = image.copy()
            if (x > 250) & (x < 550) & (y > 300) & (y < 410):
                cv2.rectangle(image, 
                            (250, 300),    # (x1, y1) Top left corner
                            (550, 410),   # (x2, y2) Bottom right corner
                            (189, 200, 135), 
                            2)
                cv2.imshow(window_name, image2)
                

            elif (x > 735) & (x < 1035) & (y > 300) & (y < 410):
                sys.exit()

    path = './exit_screen.png'
    image = cv2.imread(path)
    
    window_name = 'Exit'
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, exitOption)

    while True:
        cv2.putText(image, 
                    exit_script1,
                    (275, 200), 
                    cv2.FONT_HERSHEY_DUPLEX, 
                    2.0, 
                    (154, 162, 111), 
                    5, 
                    cv2.LINE_AA)
        
        cv2.putText(image, 
                    exit_script2,
                    (275, 260), 
                    cv2.FONT_HERSHEY_DUPLEX, 
                    2.0, 
                    (154, 162, 111), 
                    5, 
                    cv2.LINE_AA)
        
        cv2.imshow(window_name, image)
        if cv2.waitKey(0):
            break

    cv2.destroyAllWindows()
