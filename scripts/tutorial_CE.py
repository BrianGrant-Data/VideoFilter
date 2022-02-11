# CodingEntrepreneurs OpenCV Tutorial
# https://www.youtube.com/playlist?list=PLEsfXFp6DpzRyxnU-vfs3vk-61Wpt7bOS
import numpy as np
import cv2

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
    
        # Display the resulting frame
        cv2.imshow('frame',frame) # Displays the frame from cap.read()
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


    make_480p()

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read() # This is reading every frame the webcam captures
    
        # Display the resulting frame
        cv2.imshow('frame',frame) # Displays the frame from cap.read()
        if cv2.waitKey(20) & 0xFF == ord('q'): # press q in the image to exit the image
            break # Without these two lines, each time you close the image it will create another one and the loop will continue.

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


t2_rescaling()