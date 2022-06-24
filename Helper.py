# import imp
from platform import processor
import numpy as np
import cv2 as cv
from easyocr import Reader
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Input
from tensorflow.keras.optimizers import Adam
from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor

def get_reward(action,text,index):
    text = [ord(x) - 96 for x in text]
    if action+1 == text[index]:
        reward = 1
    else:
        reward = 0
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
    
def drawbbox(img,bbox):
    # Draw the bounding boxes on the (copied) input image:
    img = cv.cvtColor(img,cv.COLOR_GRAY2RGB)
    for i in range(len(bbox)):
        color = (0, 255, 0)
        cv.rectangle(img, (int(bbox[i][0]), int(bbox[i][1])), 
                (int(bbox[i][0] + bbox[i][2]), int(bbox[i][1] + bbox[i][3])), color, 2)
    return img

def ocr(img,langs,gpu):
    # OCR the input using easyocr
    print('[INFO] OCR input image---')
    reader = Reader(langs,gpu=gpu)
    results = reader.readtext(img,decoder='beamsearch',
                                min_size=0,
                                width_ths=0.01)
    return results

def build_model(states,actions):
    # print(f"model: {states}")
    model = Sequential()
    model.add(Conv2D(32,(3,3), activation='relu',input_shape=states,data_format="channels_last"))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(64,(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(64,(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(24, activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(actions, activation='linear'))

    return model
    
class CustomProcessor(Processor):
    '''
    acts as a coupling mechanism between the agent and the environment
    '''
    def process_state_batch(self, batch):
        batch = np.array(batch, dtype=object)
        batch = np.squeeze(batch,axis=1)
        return batch

processor = CustomProcessor()
def build_agent(model, actions):
    policy = BoltzmannQPolicy()
    memory = SequentialMemory(limit=50000, window_length=1)
    dqn = DQNAgent(model=model, memory=memory, policy=policy,processor=processor,
                  nb_actions=actions, nb_steps_warmup=10, target_model_update=1e-2)
    return dqn

    