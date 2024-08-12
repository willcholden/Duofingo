import cv2
# import mediapipe as mp
# from google.protobuf.json_format import MessageToDict

# cap = cv2.VideoCapture(0)

# mp_hands = mp.solutions.hands
# mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles
# hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

path = './logo.png'

# Reading an image in default mode
image = cv2.imread(path)

# Window name in which image is displayed
window_name = 'image'

# Using cv2.imshow() method
# Displaying the image

cv2.rectangle(image, 
    (40, 300),    # (x1, y1) Top left corner
    (240, 550),   # (x2, y2) Bottom right corner
    (189, 200, 135), 
    2)

cv2.rectangle(image, 
    (325, 300),    # (x1, y1) Top left corner
    (525, 550),   # (x2, y2) Bottom right corner
    (189, 200, 135), 
    2)

cv2.rectangle(image, 
    (540, 300),    # (x1, y1) Top left corner
    (740, 550),   # (x2, y2) Bottom right corner
    (189, 200, 135), 
    2)

cv2.rectangle(image, 
    (780, 300),    # (x1, y1) Top left corner
    (980, 550),   # (x2, y2) Bottom right corner
    (189, 200, 135), 
    2)

cv2.rectangle(image, 
    (1020, 300),    # (x1, y1) Top left corner
    (1220, 550),   # (x2, y2) Bottom right corner
    (189, 200, 135), 
    2)

cv2.rectangle(image, 
                      (0, 645), (1280, 720), (209, 222, 150), -1) 

cv2.imshow(window_name, image)

cv2.waitKey(0)

# closing all open windows
cv2.destroyAllWindows()
# while True:

    # ret, frame = cap.read()
    # height, width = frame.shape[:2]

    # frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # results = hands.process(frame_rgb)

    # cv2.rectangle(frame, 
    #     (200, 240),    # (x1, y1) Top left corner
    #     (440, 480),   # (x2, y2) Bottom right corner
    #     (0,0,250), 
    #     4)
            
    # cv2.rectangle(frame, 
    #     (840, 240), 
    #     (1080, 480), 
    #     (0,0,250), 
    #     4)
    
    # if results.multi_hand_landmarks:
    #     for hand_landmarks in results.multi_hand_landmarks:
    #         mp_drawing.draw_landmarks(frame,
    #                                   hand_landmarks,
    #                                   mp_hands.HAND_CONNECTIONS,
    #                                   mp_drawing_styles.get_default_hand_landmarks_style(),
    #                                   mp_drawing_styles.get_default_hand_connections_style())
            
            
            # handedness = results.multi_handedness

            # if len(handedness) == 1:
            #     hand_handedness = enumerate(handedness)
            #     handedness_dict = MessageToDict(handedness[0])
                # print(handedness_dict['classification'][0]['label'])

            

            # cv2.putText(frame, 
            #         str(type(handedness)),
            #         (20, 100), 
            #         cv2.FONT_HERSHEY_SIMPLEX, 
            #         1.3, 
            #         (230, 230, 250), 
            #         3, 
            #         cv2.LINE_AA)


    # cv2.imshow('frame', frame)
    # cv2.waitKey(50)





# cap.release()
# cv2.destroyAllWindows()


