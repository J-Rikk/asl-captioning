# Integrated Visual-Based ASL Captioning in Videoconferencing Using CNN

## Description
The client-only Automatic Sign Language Recognition (ASLR) program captures live video and processes it to determine hand presence and location. The gesture within the bounding box is then parsed by the CNN model and the closest American Sign Language word or letter is reported along with the accuracy level. The program also features a letter bank located at the bottom of the screen. The ten most recent letters are stored in the letter bank.

For region extraction and hand tracking, Google’s MediaPipe machine learning framework was chosen since it is well-documented and open-source. There are two versions of the CNN architecture used for sign language translation: MNIST and AlexNet. The MNIST model was trained on the MNIST sign language dataset and is more lightweight compared to the AlexNet model. AlexNet is one of the most popular CNN architectures and is ideal for more complex image analysis. However, this model is more resource-demanding of the two.

Though AI projects are becoming more accessible, custom training models still require a lot of computing power. Using more sophisticated models to include two-handed and dynamic gestures is one of the possible next steps to increase accessibility for sign language users in virtual spaces.

## How to Run The Program
1. Install Python by downloading the installer in your web browser through the [official Python website](https://www.python.org/downloads/windows/).
2. Download the necessary libraries for the program to execute. 
```
import pyvirtualcam
import cv2
import mediapipe as mp
import numpy as np
import keras
import time
```
3. Install **[OBS Studio](obsproject.com)** and its **[Virtual Camera Plug-in](https://obsproject.com/forum/resources/obs-virtualcam.949/)** in the **same directory**. Make sure that it is the only virtual camera software on the device.
4. Download the **ASLR_with_[model].py** with the chosen model: either _mnist_model.py_ or _alexnet_model.py_. 
5. Change the **model directory** in the program on where it is located on the device through the _model_directory_ variable. 
6. Open **OBS Studio**. Navigate to _Sources_ > _+_ (Add) > Check _Make Sources Available_ > _OK_ > _Device_ > **OBS Virtual Camera** > _OK_. A [Youtube tutorial](https://youtu.be/fkKC1uSFeCo) can be followed. 
7. Run the **ASLR_with_[model].py**. It will notify what virtual cam software it accessed for the output. 
8. Navigate to **OBS Studio** to see program execution. 
9. Navigate to chosen videoconferencing services: _Zoom_, _Google Meet_, _Discord_, _Jitsi_, etc. Set _camera output_ to **OBS Virtual Camera**.

## How to Install CUDA and CuDNN (optional)
1. Check if your device has **CUDA-enabled GPU** (only available from _NVIDIA_). A [Medium post](https://medium.com/analytics-vidhya/cuda-toolkit-on-windows-10-20244437e036) can be followed. 
2. If CUDA-enabled, download the necessary **CUDA toolkit** version on your device: [CUDA Toolkit 11.7](https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64) or [older versions](https://developer.nvidia.com/cuda-toolkit-archive).
3. If CUDA-enabled, download the necessary **CuDNN library** version on your device: [CuDNN library 8.4.1](https://developer.nvidia.com/rdp/cudnn-download) or [older versions](https://developer.nvidia.com/rdp/cudnn-archive). Refer to the [Youtube tutorial](https://www.youtube.com/watch?v=hHWkvEcDBO0&t=342s) for full installation.

## How to Install Zlib (optional)
1. Download **[Zlib](http://www.winimage.com/zLibDll/zlib123dllx64.zip)**.
2. Extract the files on your desired folder. 
3. Add the directory path of **zlibwapi.dll** to _PATH_.
4. Restart your computer. Run Python file as usual.

## Program Features
* **Webcam Dimensions:** The dimensions of the webcam must be set in the code to reflect in the output of the virtual camera. The default values for width and height are [1280, 720] pixels respectively.
* **Padding:** The 21 keypoints of MediaPipe Hands do not encapsulate the whole hand and the end points are only close to the fingertips. Padding is the added length and width of the square region to capture the whole hand. The default value of padding is 36 pixels.
* **Bounding Box Region:** Once the hand is detected, the hand is bounded by a yellow square indicating the region that is processed by the model. This region includes the padding to fit the hand inside the box. 
* **Current Prediction:** If a hand is detected, the model predicts the hand gesture to its equivalent letter or phrase. The current predicted letter or phrase, along with its accuracy, is displayed on the lower center above the letter bank in yellow text. 
* **Frame Slice and Threshold:** The frame slice stores up to _n_ predicted letters or words before it can be displayed on the letter bank. The number of occurrence of the most frequent letter in the frame slice must be greater than or equal to the threshold so that the most frequent letter can be added to the letter bank. The default value for frame slice is 15 predictions, and the threshold is 60\%. The frame slice is cleared once there is no hand detected.
* **Letter Bank:** The letter bank and stores the most frequent letters from a given frame slice and threshold. It can hold up to 10 letters (including spaces) by default and is displayed at the lower center of the screen. It is cleared in the following conditions: if it exceeds the maximum value before a new letter has been added or if no hand is detected after 30 seconds by default. 
* **Delete Function:** Exclusive to AlexNet model, the delete function is applied if the DELETE gesture is the most frequent letter in the frame slice given the threshold. This deletes the last letter of the letter bank.
## Credits
* [MNIST Model](https://github.com/chenson2018/APM-Project/blob/master/Final%20Materials/Static_Signs.ipynb?fbclid=IwAR1l7eApNeIa1lXFTH69hKjKG_qFd_WIacZY3FXmvuffWzT3zvx0IUcBEf8)
* [Alexnet Model](https://github.com/vagdevik/American-Sign-Language-Recognition-System/tree/master/2_AlexNet)