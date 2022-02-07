#%% Import Libraries
import cv2 as cv
import numpy as np
import urllib
import io

#%% Load Test Image
## Image was downloaded from here 'https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/lena.jpg'

#path = "lena.jpg"
img_needle = cv.imread("lena.jpg", cv.IMREAD_UNCHANGED)
img_haystack = cv.imread("lena_screenshot.jpg", cv.IMREAD_UNCHANGED)

result = cv.matchTemplate(img_haystack, img_needle, cv.TM_CCOEFF_NORMED)

cv.imshow('Result', result)
cv.waitKey()
# %%
