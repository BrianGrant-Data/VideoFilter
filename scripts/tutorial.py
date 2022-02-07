#%% Import Libraries
import cv2 as cv
import numpy as np
import urllib
import io

#%% Load Test Image
## Image was downloaded from here 'https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/lena.jpg'

#path = "lena.jpg"
img_needle = cv.imread("../input/lena.jpg", cv.IMREAD_UNCHANGED) # this imports the image we'll be looking for
img_haystack = cv.imread("../input/lena_screenshot.jpg", cv.IMREAD_UNCHANGED) # this imports the image we'll be searching for the small image

result = cv.matchTemplate(img_haystack, img_needle, cv.TM_CCOEFF_NORMED) # this returns an image where the white points are where the image we're looking for overlaps the background searched

#%% Show Results
cv.imshow('Result', result)
cv.waitKey()
cv.destroyAllWindows()
    
#%% Find Location of Match
min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result) # Returns the best match's position
print('Best match top left position: %s' % str(max_loc))
print('Best match confidence: %s' % max_val)

#%% Basic Thresholding 
threshold = 0.8 # Selects what confidence value we're hoping the match will exceed
if max_val >= threshold:
    print('Needle was found.')
else:
    print('Needle was not found.')

#%% Basic Bounding Box 
threshold = 0.8 # Selects what confidence value we're hoping the match will exceed
if max_val >= threshold:    
    # Get needle image dimensions
    needle_w = img_needle.shape[1]
    needle_h = img_needle.shape[0]

    # Set Bounding Box Parameters
    top_left = max_loc
    bottom_right = (top_left[0] + needle_w, top_left[1] + needle_h)

    cv.rectangle(img_haystack, top_left, bottom_right,
        color=(0,255,0), thickness=2, lineType=cv.LINE_4)

    # Display Image with Bounding Box
    cv.imshow('Result', img_haystack)
    cv.waitKey()
    cv.destroyAllWindows()
else:
    print('Conf. Threshold is too low to draw relevant bounding box.')

# %% Saving an Image as an Output
path = 'C:/Users/bag20/projects/video_filter/output/'
cv.imwrite(path + 'result.jpg', img_haystack)

# %%
