# Import Libraries
import cv2 as cv
import numpy as np
import urllib
import io

# Load Test Image
## Image was downloaded from here 'https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/lena.jpg'

#path = "lena.jpg"
img_needle = cv.imread("lena.jpg", cv.IMREAD_UNCHANGED) # this imports the image we'll be looking for
img_haystack = cv.imread("lena_screenshot.jpg", cv.IMREAD_UNCHANGED) # this imports the image we'll be searching for the small image

result = cv.matchTemplate(img_haystack, img_needle, cv.TM_CCOEFF_NORMED) # this returns an image where the white points are where the image we're looking for overlaps the background searched

# cv.imshow('Result', result)
# cv.waitKey()

min_val