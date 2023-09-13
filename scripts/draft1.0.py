# Import Libraries
import cv2 as cv
import numpy as np
import time
import mediapipe as mp
import mss
import urllib
import io
from PIL import Image, ImageGrab

# Cascade Paths
cascade_path = "C:/Users/bag20/projects/video_filter/.venv/lib/site-packages/cv2/data/" # to find your openCV package location type 'python' in the terinal and then type 'print(cv2.__file__)' in the terminal. Exit py typing 'exit()'.
face_cascade = cv.CascadeClassifier(cascade_path+'haarcascade_frontalface_alt2.xml')
# eye_cascade = cv.CascadeClassifier(cascade_path+'haarcascade_eye.xml')
# smile_cascade = cv.CascadeClassifier(cascade_path+'yyhaarcascade_smile.xml')
# uprbody_cascade = cv.CascadeClassifier(cascade_path+'haarcascade_upperbody.xml')
# profile_cascade = cv.CascadeClassifier(cascade_path+'haarcascade_profileface.xml')

# Main
def main(monitor_number):
    
    while(True):
        start = time.time()

<<<<<<< Updated upstream
        screenshot = pyautogui.screenshot() # or screenshot = ImageGrab.grab()
=======
        # Capture the Screen
        screenshot = ImageGrab.grab()

        # Prep the image
>>>>>>> Stashed changes
        screenshot = np.array(screenshot)
        screenshot = cv.cvtColor(screenshot, cv.COLOR_RGB2BGR) # np.array changes the color scheme to RGB but cv2 uses BGR and the colors will look weird if you don't switch it
        gray  = cv.cvtColor(screenshot, cv.COLOR_BGR2GRAY)

        # Pass to classifier
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5) # Returns the regions of interest
  
        # Alter the region of interest
        for (x, y, w, h) in faces:

            # Draw Bounding Box
            color = (255, 0, 0) #BGR 0-255 
            stroke = 2
            end_cord_x = x + w
            end_cord_y = y + h
            cv.rectangle(screenshot, (x, y), (end_cord_x, end_cord_y), color, stroke)

<<<<<<< Updated upstream

        # Display Frames Per Second (FPS)
=======
        # Display Time
>>>>>>> Stashed changes
        end = time.time()
        totalTime = end - start
        fps = 1 / totalTime
        cv.putText(screenshot, f'FPS: {int(fps)}', (20,70), cv.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)

        # Display image
        screenshot = cv.resize(screenshot, (1280, 720)) # Resize image # altered for first screen's aspect ration
        cv.imshow('Computer Vision', screenshot)
        
        # exit window
        if cv.waitKey(20) & 0xFF == ord('q'):
            cv.destroyAllWindows()
            break
        
    # when everything is done, release the capture
    screenshot.release()
    
if __name__ == "__main__":
    main(1)
