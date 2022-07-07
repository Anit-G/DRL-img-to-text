# import imp
from platform import processor
import numpy as np
import cv2 as cv
from easyocr import Reader
import logging
# Create and configure logger
logging.basicConfig(filename="Logs/newfile.log",
                    format='%(asctime)s %(message)s',
                    filemode='w')

# Creating an object
log = logging.getLogger()
 
# Setting the threshold of logger to DEBUG
log.setLevel(logging.DEBUG)

def convertint(bbox):
    num=[]
    for x in bbox:
        x = list(map(int,x))
        num.append(x)
    return num

def get_reward(action,text,index):
    # convert text to integers
    text = [ord(x) - 96 for x in text]
    if action+1 == text[index]:
        reward = 10
    else:
        reward = -10
    log.info(f'Reward function ({reward}): action = {action+1}, character = {text[index]}')
    return reward

def cleanup_text(text):
	# strip out non-ASCII text so we can draw the text on the image
	# using OpenCV
    text = "".join([c if ord(c) < 128 else "" for c in text]).strip()
    #convert to lower case
    text = text.lower()
    return text

def pad(img, imgh = 200,imgw=200):
    h,w = img.shape
    padded_array = np.ones([imgh,imgw], dtype=np.uint8)*255
    padded_array[0:h,0:w] = img
    return padded_array

def Sort_Tuple(tup):
    tup.sort(key = lambda x: x[0])
    return tup

def postprocess(bbox):
    bbox = Sort_Tuple(bbox)
    return bbox
    
def preprocessing(img,winsize=21,winconstant=5,maxarea = 500.0):

    # Apply the threshold:
    thresh = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, winsize, winconstant)

    # Set kernel (structuring element) size:
    kernelSize = 3
    # Set operation iterations:
    opIterations = 1
    # Get the structuring element:
    maxKernel = cv.getStructuringElement(cv.MORPH_RECT, (kernelSize, kernelSize))

    # Perform closing:
    closingImage = cv.morphologyEx(thresh, cv.MORPH_CLOSE, maxKernel, None, None, opIterations, cv.BORDER_REFLECT101)

    # character bounding boxes
    # Find the big contours/blobs on the filtered image:
    contours, hierarchy = cv.findContours(closingImage, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)

    max = 0
    for j in hierarchy[0]:
        if j[3]>max:
            max = j[3]

    contours_poly = [None] * len(contours)
    # The Bounding Rectangles will be stored here:
    boundRect = []

    # Alright, just look for the outer bounding boxes:
    for i, c in enumerate(contours):
        if hierarchy[0][i][3] == max:
            if cv.contourArea(c) > maxarea:
                contours_poly[i] = cv.approxPolyDP(c, 3, True)
                boundRect.append(cv.boundingRect(contours_poly[i]))

    return thresh,boundRect
    
def drawbbox(img,results):
    log.info(f'Drawing boxes for {results[1]}')
    # Draw the bounding boxes on the (copied) input image:
    img = cv.cvtColor(img,cv.COLOR_GRAY2RGB)
    for bbox,_,_ in results:
        color = (0, 255, 0)
        bbox = convertint(bbox)
        cv.rectangle(img, bbox[0],bbox[2], color, 2)
    return img

def ocr(img,langs,gpu):
    # OCR the input using easyocr
    log.info('[INFO] OCR input image---')
    reader = Reader(langs,gpu=gpu)
    results = reader.readtext(img,decoder='beamsearch',
                                min_size=0,
                                width_ths=0.01)
    return results
