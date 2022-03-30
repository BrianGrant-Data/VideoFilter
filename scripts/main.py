# Import Libraries
import cv2 as cv
import numpy as np
import time
import mediapipe as mp
from PIL import ImageGrab
    

def main():
    # Face mesh detection
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils

    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.3) as face_detection: # VITAL model_selection=1 because that is for close and far images and 0 is for close
        while(True):
            start = time.time()

            # Capture frame-by-frame
            image = ImageGrab.grab()
            image = np.array(image)
            image.flags.writeable = False # To improve performance, optionally mark the image as not writeable to speed up face_detection.process(image).
            results = face_detection.process(image)

            # Draw the face detection annotations on the image.
            image.flags.writeable = True # return to markable for changing the image
            image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
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
            if cv.waitKey(20) & 0xFF == ord('q'): #wait 20 milliseconds and if 'q' is pressed break the loop
                cv.destroyAllWindows()
                break
        
    # when everything is done, release the capture
    image.release()
    
if __name__ == "__main__":
    main()