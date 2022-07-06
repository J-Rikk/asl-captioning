#library for connecting to virtual camera software
import pyvirtualcam
#library for modifying video frame
import cv2
#library for hand tracking
import mediapipe as mp
#library for transforming images to array
import numpy as np
#library for loading sign language translation model
import keras
#library for checking time
import time

# import platform
# import cpuinfo
# import psutil
# import os
# import threading

import cProfile, pstats

#set the MNIST Model's location on your device
model_directory = "mnist_model.h5"

#set the dimensions of the camera (preferably your webcam's dimension)
webcam_width = 1280
webcam_height = 720

#set the additional space of the bounding box region
padding = 36
#set the number of letters stored on the letter bank
num_bank = 10
#set the number of predictions stored on frame slice
num_slice = 15 
#set the required percentage of most frequent letter on frame slice to be added on letter bank
threshold = 0.7
#set the time to clear letter bank once hand is absent
clear_time = 30
#set the text color (yellow by default)
text_color = (0, 255, 255)
#set the text font (serif font by default)
font = cv2.FONT_HERSHEY_TRIPLEX

#classes available on the MNIST model in order
letters = "ABCDEFGHIKLMNOPQRSTUVWXY"

#loads the MNIST model
model = keras.models.load_model(model_directory)

# cpu_model = cpuinfo.get_cpu_info()["brand_raw"]
# ram_total = psutil.virtual_memory()[0]

def clear_console():
    command = 'cls' if os.name in ('nt', 'dos') else 'clear'
    os.system(command)

def usage_report(cpu_model, ram_total):
    open('cpu_log.txt','w').close()
    open('ram_log.txt','w').close()

    while True:
        cpu_log = open('cpu_log.txt','a')
        ram_log = open('ram_log.txt','a')

        time.sleep(1)

        ram_percent = psutil.virtual_memory()[2]
        ram_used = psutil.virtual_memory()[3]

        cpu_log.write(f'{psutil.cpu_percent()}\n')
        ram_log.write(f'{round(ram_used/10**9, 2)}\n')

        clear_console()

        print(f'CPU Usage: {psutil.cpu_percent()}% ({cpu_model})')
        print(f'RAM Usage: {ram_percent}% ({round(ram_used/10**9, 2)}/{round(ram_total/10**9,2)} GB)')

        cpu_log.close()
        ram_log.close()

#function to make the pixel points into even and to add the set padding
def make_even_put_padding(coord):
    for x in range(2):
        for y in range(2):
            if(coord[x][y]%2 != 0): #make the points even
                coord[x][y] += 1
            if(y == 0):
                if((coord[x][y] - padding) > 0): #does not add padding if additional space goes over the camera boundaries (left, top)
                    coord[x][y] -= padding
            else:
                if((coord[x][y] + padding) < coord[x][2]): #does not add padding if additional space goes over the camera boundaries (right, bottom)
                    coord[x][y] += padding
    return coord

#function to check if the final points went past the camera boundaries
def out_of_bounds(coord):
    for x in range(2):
        for y in range(2):
            if(y == 0):
                if(coord[x][y] < 0): #checks the left and top side of the region
                    return True
            else:
                if(coord[x][y] > coord[x][2]): #checks the right and bottom side of the region
                    return True
    return False

#function to find the letter with the highest number of occurrence on frame slice
def most_frequent_letter(frame_slice):
    return max(set(frame_slice), key = frame_slice.count)

#function to find the number of occurrence of most frequent letter on frame slice
def num_of_occurrence(frame_slice, max_letter):
    return frame_slice.count(max_letter)

#function to centralize text on video frame
def center_text(width, text, font):
    text_size = cv2.getTextSize(text, font, 1, 2)[0]
    return int((width - text_size[0]) / 2)

def region_extraction(hand_landmarks, h, w):
    for handLMs in hand_landmarks:
        x_max = 0
        y_max = 0
        x_min = w
        y_min = h
        for lm in handLMs.landmark:
            x, y = int(lm.x * w), int(lm.y * h)
            if x > x_max:
                x_max = x
            if x < x_min:
                x_min = x
            if y > y_max:
                y_max = y
            if y < y_min:
                y_min = y

            coord = [[x_min, x_max, w], [y_min, y_max, h]]
            #print(coord)
            padding = 24
            for x in range(2):
                for y in range(2):
                    if(coord[x][y]%2 != 0):
                        coord[x][y] += 1
                    if(y == 0):
                        if((coord[x][y] - padding) > 0):
                            coord[x][y] -= padding
                    else:
                        if((coord[x][y] - padding) < coord[x][2]):
                            coord[x][y] += padding

            xd = coord[0][1] - coord[0][0] 
            yd = coord[1][1]  - coord[1][0] 

            dist = [xd, yd]
            diff = max(dist) - min(dist)
            diff_div = int(diff/2)

            smaller = dist.index(min(dist))

            end_bound = coord[smaller][2] - coord[smaller][1]

            coord[smaller][0] -= diff_div
            coord[smaller][1] += diff_div

    return coord

def mnist_predict(resized, letters, model):
    prediction_array = model.predict(np.expand_dims(np.expand_dims(resized, axis = 2), axis = 0))
    prediction = np.argmax(prediction_array, axis = 1)[0]
    percentage = np.amax(prediction_array)

    #retrieves the letter through the index with the highest probability
    letter = letters[prediction]

    #produces string with predicted letter and accuracy
    letter_w_percent = letter  + " " + "{:.0%}".format(percentage)

    return letter, letter_w_percent

def sys(model, letters):
    cap = cv2.VideoCapture(0)

    # set new dimensions to camera object
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, webcam_width)  
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, webcam_height)

    #set the color format of pyvirtual cam (RGB by default), BGR is the default format of OpenCV
    fmt = pyvirtualcam.PixelFormat.BGR

    # t1 = threading.Thread(target=usage_report, args=[cpu_model, ram_total])
    # t1.start()

    #set the initial time for previous frame when the program runs
    old_time = 0
    predict_calls = 0
    max_predict = 50

    with pyvirtualcam.Camera(width=webcam_width, height=webcam_height, fps=20, fmt=fmt) as cam:
        #prints the software accessed for web camera (OBS by default)
        print(f'Using virtual camera: {cam.device}')
        
        while predict_calls < max_predict:
            #variables needed for hand tracking
            mphands = mp.solutions.hands
            hands = mphands.Hands()
            mp_drawing = mp.solutions.drawing_utils
            hand_present = False

            #reads the current video frame
            _, frame = cap.read()
            h, w, c = frame.shape
            frame_slice = []
            letter_bank = ""

            while predict_calls < max_predict:
                #reads the current video frame
                _, frame = cap.read()
                #flips the current video frame
                frame = cv2.flip(frame, 1)
                framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = hands.process(framergb)
                hand_landmarks = result.multi_hand_landmarks
                
                #checks if the hand is present on the screen
                if hand_landmarks:
                    coord = region_extraction(hand_landmarks, h, w)
                    # for handLMs in hand_landmarks:
                    #     x_max = 0
                    #     y_max = 0
                    #     x_min = w
                    #     y_min = h

                    #     #looks for the minimum and maximum values for x and y in 21-point hand landmarks
                    #     for lm in handLMs.landmark:
                    #         x, y = int(lm.x * w), int(lm.y * h)
                    #         if x > x_max:
                    #             x_max = x
                    #         if x < x_min:
                    #             x_min = x
                    #         if y > y_max:
                    #             y_max = y
                    #         if y < y_min:
                    #             y_min = y

                    # #stores the x and y (with width and height of camera boundaries) separately into a list
                    # coord = [[x_min, x_max, w], [y_min, y_max, h]]
                    # coord = make_even_put_padding(coord)

                    # #calculates for the distance of maximum value and minimum value of x
                    # xd = coord[0][1] - coord[0][0]
                    # #calculates for the distance of maximum value and minimum value of y 
                    # yd = coord[1][1]  - coord[1][0] 
                    # #stores the distances to a list
                    # dist = [xd, yd]
                    # #gets the difference of maximum and minimum distances
                    # diff = max(dist) - min(dist)
                    # #divides the difference by 2, must be an integer due to pixels
                    # diff_div = int(diff/2)

                    # #gets the index of minimum distances on distance list
                    # smaller = dist.index(min(dist))

                    # #distributes the difference to either left or top side of the region
                    # coord[smaller][0] -= diff_div
                    # #distributes the difference to either right or bottom side of the region
                    # coord[smaller][1] += diff_div

                    #does not proceed with prediction if final coordinates are out of bounds
                    if(out_of_bounds(coord)):
                        continue
                    else:
                        #adds the colored square in the video frame
                        cv2.rectangle(frame, (coord[0][0], coord[1][0]), (coord[0][1], coord[1][1]), text_color, 2)
                        #extracts the region of interest (square)
                        roi = frame[coord[1][0]:coord[1][1], coord[0][0]:coord[0][1]]
                        #resizes the extracted ROI to 28x28 image as needed by the MNIST model
                        resized = cv2.resize(roi, (28,28), interpolation=cv2.INTER_AREA)
                        #recolors the extracted ROI to graysclae as needed by the MNIST model
                        resized = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
                        letter, letter_w_percent = mnist_predict(resized, letters, model)
                        predict_calls += 1

                        # #produces the list of predictions by the model
                        # prediction_array = model.predict(np.expand_dims(np.expand_dims(resized, axis = 2), axis = 0))
                        # #retrieves the index of the class / letter with the highest probability predicted by the model
                        # prediction = np.argmax(prediction_array, axis = 1)[0]
                        # #retrieves the highest probability of the class
                        # percentage = np.amax(prediction_array)

                        # #retrieves the letter through the index with the highest probability
                        # letter = letters[prediction]

                        # #produces string with predicted letter and accuracy
                        # letter_w_percent = letter  + " " + "{:.0%}".format(percentage)
                        #gets the x-coordinate to centralize predicted letter and accuracy
                        letter_center = center_text(w, letter_w_percent, font)
                        #adds the letter / class to frame slice
                        frame_slice.append(letter)
                        #displays the centered predicted letter and accuracy in video frame
                        cv2.putText(frame, letter_w_percent, (letter_center, 600), cv2.FONT_HERSHEY_TRIPLEX, 2.0, text_color, thickness=4)

                        #checks if the frame slice is equal to or greater than set number of slice
                        if len(frame_slice) >= num_slice:
                            #retrieves the most frequent letter in frame slice
                            max_letter = most_frequent_letter(frame_slice)
                            #checks if the most frequent letter occurrence is equal or greater than the threshold
                            if((num_of_occurrence(frame_slice, max_letter)/num_slice) >= threshold):
                                #clears letter bank if it exceeds or reaches the maximum number of letters
                                if(len(letter_bank) >= num_bank):
                                    letter_bank = ""
                                #adds the most frequent letter in letter bank
                                letter_bank += max_letter
                            #clears the frame slice
                            frame_slice.clear()
                        #changes boolean to True since hand is present
                        hand_present = True

                else:
                    #clears frame slice if hand is absent on the screen
                    frame_slice.clear()
                    if(hand_present == True):
                        #gets the time of last frame processed
                        old_time = time.time()
                    #changes boolean to False since hand is absent
                    hand_present = False

                #gets x-coordinate to centralize letter bank
                bank_center = center_text(w, letter_bank, font)
                #displays the centered letter bank in video frame
                cv2.putText(frame, letter_bank, (bank_center, 670), cv2.FONT_HERSHEY_TRIPLEX, 2.0, text_color, thickness=4)

                #clears letter bank after clear time once hand is absent
                if((time.time() - old_time) > clear_time) and (not hand_present):
                    letter_bank = ""

                #sends video frame to virtual camera
                cam.send(frame)
                cam.sleep_until_next_frame()

cProfile.run('sys(model, letters)',"output.dat")

with open('mnist_time_profiling.txt','w') as f:
    p = pstats.Stats("output.dat",stream=f)
    p.sort_stats('time').print_stats()