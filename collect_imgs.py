import os
import cv2 
import time 

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

dataset_size = 100
cap = cv2.VideoCapture(0)


while True:

    
    item = input("Enter [category]/[item] OR Enter 'quit' to quit: ")
    if item == "quit":
        break

    if not os.path.exists(os.path.join(DATA_DIR, item)):
        os.makedirs(os.path.join(DATA_DIR, item)) 

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

    counter = 0
    while counter < dataset_size:
        str2 = "Recording: " + item 
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
                    str2, 
                    (100, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1.3, 
                    (0, 255, 0), 
                    3, 
                    cv2.LINE_AA)
        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(DATA_DIR, item, '{}.jpg'.format(counter)), frame)

        counter += 1

cap.release()
cv2.destroyAllWindows()