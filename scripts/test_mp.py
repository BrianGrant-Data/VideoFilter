# Tutorial from "The Coding Lib" https://www.youtube.com/watch?v=K4nn4YdSMFM&list=WL&index=4&t=550s
# This tutorial will check to see if the mediapipe package can improve the speed of or face detection program

import cv2
import mediapipe as mp 
import time

def tutorial_facemesh():
    '''Tutorial from "The Coding Lib" https://www.youtube.com/watch?v=K4nn4YdSMFM&list=WL&index=4&t=550s'''

    # Face mesh detection
    mp_face_mesh = mp.solutions.face_mesh

    mp_drawing = mp.solutions.drawing_utils
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)



    cap = cv2.VideoCapture(0)

    with mp_face_mesh.FaceMesh(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:


        while cap.isOpened():

            success, image = cap.read()

            start = time.time()

            # Flip the image horizontally for a later selfie-view display
            # Convert the BGR image to RGB.
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False

            # Process the image
            results = face_mesh.process(image)

            image.flags.writeable = True

            # Convert the image color back so it can be displayed
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    #print(face_landmarks)
                    #print(face_landmarks.landmark.x)
                    # Draw the face mesh annotations on the image.
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=drawing_spec,
                        connection_drawing_spec=drawing_spec)



            end = time.time()
            totalTime = end - start

            fps = 1 / totalTime

            cv2.putText(image, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)

            cv2.imshow('MediaPipe FaceMesh', image)



            if cv2.waitKey(5) & 0xFF == ord('q'): # hit q to escape
                break

    cap.release()

def tutorial_facedetection():
    '''
    Appling mediapipe.solutions.face_detection instead of face_mesh
    https://google.github.io/mediapipe/solutions/face_detection
    '''

    # Face mesh detection
    mp_face_detection = mp.solutions.face_detection
    
    mp_drawing = mp.solutions.drawing_utils
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

    # For webcam input:
    cap = cv2.VideoCapture(0)
    with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
        while cap.isOpened():
            success, image = cap.read()
            start = time.time()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_detection.process(image)

            # Draw the face detection annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.detections:
                for detection in results.detections:
                    mp_drawing.draw_detection(image, detection)

            # Flip the image horizontally for a selfie-view display.
            image = cv2.flip(image, 1)

            # Display FPS on image
            end = time.time()
            totalTime = end - start
            if totalTime == 0: 
                fps = 999 
            else:
                fps = 1 / totalTime

            cv2.putText(image, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
            
            # Display image
            cv2.imshow('MediaPipe Face Detection', image)

            if cv2.waitKey(5) & 0xFF == ord('q'): # hit q to escape
                break

    cap.release()


def main():
    import cv2 as cv
    import numpy as np
    import time
    import mediapipe as mp
    import mss
    import urllib
    import io
    from PIL import Image, ImageGrab
    import pyautogui
    
    # Face mesh detection
    mp_face_detection = mp.solutions.face_detection
    
    mp_drawing = mp.solutions.drawing_utils
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

    with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
        while(True):
            start = time.time()

            # Capture frame-by-frame
            image = pyautogui.screenshot() # or screenshot = ImageGrab.grab()
            # # screenshot = ImageGrab.grab()
            image = np.array(image)
            image = cv.cvtColor(image, cv.COLOR_RGB2BGR) # np.array changes the color scheme to RGB but cv2 uses BGR and the colors will look weird if you don't switch it
            
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_detection.process(image)

            # Draw the face detection annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.detections:
                for detection in results.detections:
                    mp_drawing.draw_detection(image, detection)


            # Display Time
            end = time.time()
            totalTime = end - start
            fps = 1 / totalTime

            cv.putText(image, f'FPS: {int(fps)}', (20,70), cv.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
            cv.imshow('Computer Vision', image)
            
            # exit window
            if cv.waitKey(20) & 0xFF == ord('q'):
                cv.destroyAllWindows()
                break
        
    # when everything is done, release the capture
    image.release()


main()