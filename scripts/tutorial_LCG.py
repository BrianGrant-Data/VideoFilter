# Learn Code By Gaming's OpenCV Tutorial

# Import Libraries
import cv2 as cv
import numpy as np
import urllib
import io
import sys
from time import time
from PIL import Image, ImageGrab
import pyautogui
import win32gui, win32ui, win32con

class WindowCapture:

    # properties
    w = 0
    h = 0
    hwnd = None
    cropped_x = 0
    cropped_y = 0
    offset_x = 0
    offset_y = 0

    # constructor
    def __init__(self, window_name=None):
        # find the handle for the window we want to capture
        if window_name is None:
            self.hwnd = win32gui.GetDesktopWindow()
        else:
            self.hwnd = win32gui.FindWindow(None, window_name)
            if not self.hwnd:
                raise Exception('Window not found: {}'.format(window_name))

        # get the window size
        window_rect = win32gui.GetWindowRect(self.hwnd)
        self.w = window_rect[2] - window_rect[0]
        self.h = window_rect[3] - window_rect[1]

        # account for the window border and titlebar and cut them off
        border_pixels = 8
        titlebar_pixels = 30
        self.w = self.w - (border_pixels * 2)
        self.h = self.h - titlebar_pixels - border_pixels
        self.cropped_x = border_pixels
        self.cropped_y = titlebar_pixels

        # set the cropped coordinates offset so we can translate screenshot
        # images into actual screen positions
        self.offset_x = window_rect[0] + self.cropped_x
        self.offset_y = window_rect[1] + self.cropped_y

    def get_screenshot(self):

        # get the window image data
        wDC = win32gui.GetWindowDC(self.hwnd)
        dcObj = win32ui.CreateDCFromHandle(wDC)
        cDC = dcObj.CreateCompatibleDC()
        dataBitMap = win32ui.CreateBitmap()
        dataBitMap.CreateCompatibleBitmap(dcObj, self.w, self.h)
        cDC.SelectObject(dataBitMap)
        cDC.BitBlt((0, 0), (self.w, self.h), dcObj, (self.cropped_x, self.cropped_y), win32con.SRCCOPY)

        # convert the raw data into a format opencv can read
        #dataBitMap.SaveBitmapFile(cDC, 'debug.bmp')
        signedIntsArray = dataBitMap.GetBitmapBits(True)
        img = np.fromstring(signedIntsArray, dtype='uint8')
        img.shape = (self.h, self.w, 4)

        # free resources
        dcObj.DeleteDC()
        cDC.DeleteDC()
        win32gui.ReleaseDC(self.hwnd, wDC)
        win32gui.DeleteObject(dataBitMap.GetHandle())

        # drop the alpha channel, or cv.matchTemplate() will throw an error like:
        #   error: (-215:Assertion failed) (depth == CV_8U || depth == CV_32F) && type == _templ.type() 
        #   && _img.dims() <= 2 in function 'cv::matchTemplate'
        img = img[...,:3]

        # make image C_CONTIGUOUS to avoid errors that look like:
        #   File ... in draw_rectangles
        #   TypeError: an integer is required (got type tuple)
        # see the discussion here:
        # https://github.com/opencv/opencv/issues/14866#issuecomment-580207109
        img = np.ascontiguousarray(img)

        return img

    # find the name of the window you're interested in.
    # once you have it, update window_capture()
    # https://stackoverflow.com/questions/55547940/how-to-get-a-list-of-the-name-of-every-open-window
    @staticmethod
    def list_window_names():
        def winEnumHandler(hwnd, ctx):
            if win32gui.IsWindowVisible(hwnd):
                print(hex(hwnd), win32gui.GetWindowText(hwnd))
        win32gui.EnumWindows(winEnumHandler, None)

    # translate a pixel position on a screenshot image to a pixel position on the screen.
    # pos = (x, y)
    # WARNING: if you move the window being captured after execution is started, this will
    # return incorrect coordinates, because the window position is only calculated in
    # the __init__ constructor.
    def get_screen_position(self, pos):
        return (pos[0] + self.offset_x, pos[1] + self.offset_y)

def tutorial1():
    # Load Test Image
    ## Image was downloaded from here 'https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/lena.jpg'

    #path = "lena.jpg"
    img_needle = cv.imread("../input/lena.jpg", cv.IMREAD_UNCHANGED) # this imports the image we'll be looking for
    img_haystack = cv.imread("../input/lena_screenshot.jpg", cv.IMREAD_UNCHANGED) # this imports the image we'll be searching for the small image

    result = cv.matchTemplate(img_haystack, img_needle, cv.TM_CCOEFF_NORMED) # this returns an image where the white points are where the image we're looking for overlaps the background searched

    # Show Results
    cv.imshow('Result', result)
    cv.waitKey()
    cv.destroyAllWindows()
    
def tutorial2():
    # Load Test Image
    ## Image was downloaded from here 'https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/lena.jpg'

    #path = "lena.jpg"
    img_needle = cv.imread("../input/lena.jpg", cv.IMREAD_UNCHANGED) # this imports the image we'll be looking for
    img_haystack = cv.imread("../input/lena_screenshot.jpg", cv.IMREAD_UNCHANGED) # this imports the image we'll be searching for the small image

    result = cv.matchTemplate(img_haystack, img_needle, cv.TM_CCOEFF_NORMED) # this returns an image where the white points are where the image we're looking for overlaps the background searched

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

def tutorial2a():
    # Load Test Image
    ## Image was downloaded from here 'https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/lena.jpg'

    #path = "lena.jpg"
    img_needle = cv.imread("../input/lena.jpg", cv.IMREAD_UNCHANGED) # this imports the image we'll be looking for
    img_haystack = cv.imread("../input/lena_screenshot.jpg", cv.IMREAD_UNCHANGED) # this imports the image we'll be searching for the small image

    result = cv.matchTemplate(img_haystack, img_needle, cv.TM_CCOEFF_NORMED) # this returns an image where the white points are where the image we're looking for overlaps the background searched

    # Saving an Image as an Output
    path = 'C:/Users/bag20/projects/video_filter/output/gitignore'
    cv.imwrite(path + 'result.jpg', img_haystack)

def tutorial4(method = ["pyauto", "ImageGrab", "wincap"], screen_name = "tutorial_LCG.py - video_filter - Visual Studio Code"):
    methods = ["pyauto", "ImageGrab", "wincap"]
    if method not in methods:
        sys.exit("Method not in methods")
    
    # %% displaying a screen
    loop_time = time()
    while(True):
        if method == "pyauto":
            screenshot = pyautogui.screenshot() # or screenshot = ImageGrab.grab()
        elif method == "ImageGrab":
            screenshot = ImageGrab.grab()
        elif method == "wincap":
            # initialize the WindowCapture class
            wincap = WindowCapture(str(screen_name))
            screenshot = wincap.get_screenshot()
            
        screenshot = np.array(screenshot)
        screenshot = cv.cvtColor(screenshot, cv.COLOR_RGB2BGR) # np.array changes the color scheme to RGB but cv2 uses BGR and the colors will look weird if you don't switch it

        cv.imshow('Computer Vision', screenshot)

        # Display frames per second (FPS)
        print('FPS {}'.format(round(1/(time() - loop_time)), 2)) # goal is 24 fps minimum. time()-loop_time returns the seconds per frame (SPF). 1/SPF returns the frames per second (FPS)
        loop_time = time()

        if cv.waitKey(20) & 0xFF == ord('q'):
            # screenshot.release()
            cv.destroyAllWindows()
            break

    # When everything done, release the capture
    
tutorial4("pyauto")
# tutorial4("wincap", "Real-time Object Detection - OpenCV Object Detection in Games #5 - YouTube - Google Chrome")
# WindowCapture.list_window_names()
