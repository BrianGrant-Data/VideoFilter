# Import Libraries
import cv2 as cv
import numpy as np
from PIL import Image, ImageGrab
import pyautogui

# Cascade Paths
cascade_path = "C:/Users/bag20/projects/video_filter/.venv/lib/site-packages/cv2/data/" # to find your openCV package location type 'python' in the terinal and then type 'print(cv2.__file__)' in the terminal. Exit py typing 'exit()'.
face_cascade = cv.CascadeClassifier(cascade_path+'haarcascade_frontalface_alt2.xml')

def main():
    # display screen
    while(True):
        screenshot = pyautogui.screenshot() # or screenshot = ImageGrab.grab()
        # # screenshot = ImageGrab.grab()
        screenshot = np.array(screenshot)
        screenshot = cv.cvtColor(screenshot, cv.COLOR_RGB2BGR) # np.array changes the color scheme to RGB but cv2 uses BGR and the colors will look weird if you don't switch it
        
        # Capture frame-by-frame
        gray  = cv.cvtColor(screenshot, cv.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
  
        for (x, y, w, h) in faces:
            #print(x,y,w,h)
            roi_gray = gray[y:y+h, x:x+w] #(ycord_start, ycord_end)
            roi_color = screenshot[y:y+h, x:x+w]

            # Draw Bounding Box
            color = (255, 0, 0) #BGR 0-255 
            stroke = 2
            end_cord_x = x + w
            end_cord_y = y + h
            cv.rectangle(screenshot, (x, y), (end_cord_x, end_cord_y), color, stroke)

        cv.imshow('Computer Vision', screenshot)
        
    # exit window
        if cv.waitKey(20) & 0xFF == ord('q'):
            cv.destroyAllWindows()
            break
    # when everything is done, release the capture
    screenshot.release()
    
if __name__ == "__main__":
    main()