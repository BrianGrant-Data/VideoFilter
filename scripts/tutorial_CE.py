# CodingEntrepreneurs OpenCV Tutorial
# https://www.youtube.com/playlist?list=PLEsfXFp6DpzRyxnU-vfs3vk-61Wpt7bOS

import numpy as np
import cv2
import os
import PIL
import pyautogui

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def t1_webcam_display(): # https://www.youtube.com/watch?v=YY9f-6u2Q_c&list=PLEsfXFp6DpzRyxnU-vfs3vk-61Wpt7bOS&index=3
    '''
    Simple display and altering of a webcam stream.
    '''
    cap = cv2.VideoCapture(0) # Returns video from webcams. The 0 returns from the first webcam, 1 would be from the second and so on.

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read() # This is reading every frame the webcam captures

        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
        # Display the unaltered frame
        cv2.imshow('frame',frame) # Displays the frame from cap.read()
        
        # Display the altered frame
        cv2.imshow('gray',gray) # Displays the frame after we altered it
        
        if cv2.waitKey(20) & 0xFF == ord('q'): # press q in the image to exit the image
            break # Without these two lines, each time you close the image it will create another one and the loop will continue.

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

def t2_rescaling(): # https://www.youtube.com/watch?v=y76C3P20rwc&list=PLEsfXFp6DpzRyxnU-vfs3vk-61Wpt7bOS&index=4
    '''
    Changes the resolution for either upscaling or downscaling.
    You'll mostly downscale the resolution so the hardware can handle too detailed video streams.
    '''    
    cap = cv2.VideoCapture(0)

    def rescale_frame(frame, percent=75):
        scale_percent = percent
        width = int(frame.shape[1] * scale_percent / 100)
        height = int(frame.shape[0] * scale_percent / 100)
        dim = (width, height) # dimensions
        return cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
    
    def change_res(width, height):
        cap.set(3, width) # cap.set changes parameters in the object "cap". 3 changes the width parameter, 4 changes the height parameter
        cap.set(4, height)

    def make_1080p():
        cap.set(3, 1920)
        cap.set(4, 1080)

    def make_720p():
        cap.set(3, 1280)
        cap.set(4, 720)
    
    def make_480p():
        cap.set(3, 640)
        cap.set(4, 480)

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read() # This is reading every frame the webcam captures

        # Display the unaltered frame
        cv2.imshow('frame',frame) # Displays the frame from cap.read()

        # Display the altered frame
        frame_alt = rescale_frame(frame, percent=30)
        cv2.imshow('frame_alt', frame_alt) # Displays the frame from cap.read()
        
        if cv2.waitKey(20) & 0xFF == ord('q'): # press q in the image to exit the image
            break # Without these two lines, each time you close the image it will create another one and the loop will continue.

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

# def t3_record_video(
#     filename="video.mp4",
#     fps=24.0,
#     res="720p"): 
#     # https://www.codingforentrepreneurs.com/blog/how-to-record-video-in-opencv-python/ #https://www.youtube.com/watch?v=1eHQIu4r0Bc&list=PLEsfXFp6DpzRyxnU-vfs3vk-61Wpt7bOS&index=5
#     '''
#     Default parameters:

#     filename = "video.mp4" # .mp4 and .avi are the file types focuse on here
#     fps = 24.0 # Frames per second
#     res = "720p"
#     '''
#     import numpy as np
#     import os
#     import cv2

#     # Set resolution for the video capture
#     # Function adapted from https://kirr.co/0l6qmh
#     def change_res(cap, width, height):
#         cap.set(3, width)
#         cap.set(4, height)

#     # Standard Video Dimensions Sizes
#     STD_DIMENSIONS =  {
#         "480p": (640, 480),
#         "720p": (1280, 720),
#         "1080p": (1920, 1080),
#         "4k": (3840, 2160),
#     }

#     # grab resolution dimensions and set video capture to it.
#     def get_dims(cap, res='1080p'):
#         width, height = STD_DIMENSIONS["480p"]
#         if res in STD_DIMENSIONS:
#             width,height = STD_DIMENSIONS[res]
#         ## change the current caputre device
#         ## to the resulting resolution
#         change_res(cap, width, height)
#         return width, height

#     # Video Encoding, might require additional installs
#     # Types of Codes: http://www.fourcc.org/codecs.php
#     VIDEO_TYPE = {
#         'avi': cv2.VideoWriter_fourcc(*'XVID'),
#         #'mp4': cv2.VideoWriter_fourcc(*'H264'),
#         'mp4': cv2.VideoWriter_fourcc(*'XVID'),
#     }

#     def get_video_type(filename):
#         filename, ext = os.path.splitext(filename)
#         if ext in VIDEO_TYPE:
#         return  VIDEO_TYPE[ext]
#         return VIDEO_TYPE['avi']

#     cap = cv2.VideoCapture(0)
#     out = cv2.VideoWriter(filename, get_video_type(filename), 25, get_dims(cap, res))

#     while True:
#         ret, frame = cap.read()
#         out.write(frame)
#         cv2.imshow('frame',frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     out.release()
#     cv2.destroyAllWindows()

# Tutorial 4 face recognition
def t4_face_cascadeClassifier():
    '''
    https://www.youtube.com/watch?v=PmZ29Vta7Vc&list=PLEsfXFp6DpzRyxnU-vfs3vk-61Wpt7bOS&index=7
    '''
    import numpy as np
    import cv2

    cascade_path = "C:/Users/bag20/projects/video_filter/.venv/lib/site-packages/cv2/data/" # to find your openCV package location type 'python' in the terinal and then type 'print(cv2.__file__)' in the terminal. Exit py typing 'exit()'.
    face_cascade = cv2.CascadeClassifier(cascade_path+'haarcascade_frontalface_alt2.xml')
    eye_cascade = cv2.CascadeClassifier(cascade_path+'haarcascade_eye.xml')
    smile_cascade = cv2.CascadeClassifier(cascade_path+'yyhaarcascade_smile.xml')

    cap = cv2.VideoCapture(0)

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
        for (x, y, w, h) in faces:
            #print(x,y,w,h)
            roi_gray = gray[y:y+h, x:x+w] #(ycord_start, ycord_end)
            roi_color = frame[y:y+h, x:x+w]

            # Draw Bounding Box
            color = (255, 0, 0) #BGR 0-255 
            stroke = 2
            end_cord_x = x + w
            end_cord_y = y + h
            cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)
            #subitems = smile_cascade.detectMultiScale(roi_gray)
            #for (ex,ey,ew,eh) in subitems:
            #	cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

        # Display the resulting frame
        cv2.imshow('frame',frame)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    cap.release()
    cv2.destroyAllWindows()

def t4a_face_recognizer():
    '''
    https://www.youtube.com/watch?v=PmZ29Vta7Vc&list=PLEsfXFp6DpzRyxnU-vfs3vk-61Wpt7bOS&index=7
    '''
    import numpy as np
    import cv2
    import pickle

    cascade_path = "C:/Users/bag20/projects/video_filter/.venv/lib/site-packages/cv2/data/" # to find your openCV package location type 'python' in the terinal and then type 'print(cv2.__file__)' in the terminal. Exit py typing 'exit()'.
    face_cascade = cv2.CascadeClassifier(cascade_path+'haarcascade_frontalface_alt2.xml')
    eye_cascade = cv2.CascadeClassifier(cascade_path+'haarcascade_eye.xml')
    smile_cascade = cv2.CascadeClassifier(cascade_path+'yyhaarcascade_smile.xml')

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("./recognizers/face-trainner.yml")

    labels = {"person_name": 1}
    with open("pickles/face-labels.pickle", 'rb') as f:
        og_labels = pickle.load(f)
        labels = {v:k for k,v in og_labels.items()}

    cap = cv2.VideoCapture(0)

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
        for (x, y, w, h) in faces:
            #print(x,y,w,h)
            roi_gray = gray[y:y+h, x:x+w] #(ycord_start, ycord_end)
            roi_color = frame[y:y+h, x:x+w]

            #image recognizer. You could use a deep learning model to predict as well: keras tensorflow pytorch scikit learn
            id_, conf = recognizer.predict(roi_gray)
            if conf>=4 and conf <= 85:
                #print(5: #id_)
                #print(labels[id_])
                font = cv2.FONT_HERSHEY_SIMPLEX
                name = labels[id_]
                color = (255, 255, 255)
                stroke = 2
                cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)

            img_item = "7.png"
            cv2.imwrite(img_item, roi_color)

            # Draw Bounding Box
            color = (255, 0, 0) #BGR 0-255 
            stroke = 2
            end_cord_x = x + w
            end_cord_y = y + h
            cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)
            #subitems = smile_cascade.detectMultiScale(roi_gray)
            #for (ex,ey,ew,eh) in subitems:
            #	cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        # Display the resulting frame
        cv2.imshow('frame',frame)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    cap.release()
    cv2.destroyAllWindows()



# Tutorial 5 overlay
# Tutorial 6 Third party Cascades
# Tutorial 8 Image Filters

t4_face_cascadeClassifier()
