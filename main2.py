import matplotlib.pyplot as plt
from Helper import *
from gym import Env
from gym.spaces import Discrete, Box
import random
link = 'Resources/sample.png'
langs = ['en']
gpu = False


class pdfread(Env):
	def __init__(self,link):
		self.action_space = Discrete(26)
		self.observation_space = Box(low=0, high=255, shape=(330,3300,1,))

		# start state
		self.state = np.ones([330,3300,1], dtype=np.uint8)
		self.img = cv.imread(link,cv.IMREAD_GRAYSCALE)
		# self.img = cv.imread(link)
		self.results = ocr(self.img,['en'],False)
		self.windex = 0
		self.cindex = 0
		self.c_results = 0
		

	def step(self,action):
		def characterR(result,w_index,img):
			"""
			Arguments: 
						results: the output of the ocr function which gives use the segmented bounding boxes of
						all the words in the image. Also the text in each of the segmented boxes as well as the 
						probability of the word.

						w_index: is a tracker of which word is currently under attention.
						c_index: is a tracker of which character in the word is currently under attention.
			Return:
						Based on which word and character we are on in the image/passage pass the bounding box of the character
						The image of the character and the text of the whole word
			"""
			# use the word tracker
			bbox,text,prob = result[w_index]
			# unpack the bounding box
			(tl, tr, br, bl) = bbox
			tl = (int(tl[0]), int(tl[1]))
			tr = (int(tr[0]), int(tr[1]))
			br = (int(br[0]), int(br[1]))
			bl = (int(bl[0]), int(bl[1]))

			cropped_img = cv.adaptiveThreshold(img[tl[1]:br[1], tl[0]:br[0]],255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 21,5)
			# cleanup the text in the word bounding box
			text = cleanup_text(text)
            
			# Bounding box for individual characters
			cropped_img = pad(cv.resize(cropped_img,(cropped_img.shape[1]*10,cropped_img.shape[0]*10),interpolation = cv.INTER_LINEAR),330,3300)
		
			return cropped_img,bbox,text
		# check whether the episode is finished
		if len(self.results)-1 == self.windex:
			done = True
		else:
			done = False

		w_img,wbbox,text = characterR(self.results,self.windex,self.img)	

		# define the state that is going to the model
		self.state = np.expand_dims(w_img,axis=2)
		
        #  reward based on the previous action
		reward = get_reward(action,text,self.cindex)

		#update index
		if self.cindex == len(text)-1:
			print(f'[INFO] Moving to next word, (previous index): {self.windex}')
			self.cindex = 0
			self.windex+=1
		else:	
			#update character index
			self.cindex +=1
		
		# placeholder used for diagnosis may contain probabilites
		info = {}

		return self.state, reward, done, info

	def render(self):
		# i don't need to render any visual
		pass

	def reset(self):
		# reset states
		self.state = np.ones([330,3300,1], dtype=np.uint8)
		self.img = cv.imread(link,cv.IMREAD_GRAYSCALE)
		self.results = ocr(self.img,['en'],False)
		self.windex = 0
		self.cindex = 0
		self.c_results = 0
		return self.state

env = pdfread(link=link)
states = env.observation_space.shape
actions = env.action_space.n

print(f"Size of states: {states}")
print(f"Number of actions: {actions}")

model = build_model(states,actions)
model.summary()

dqn = build_agent(model, actions)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])
dqn.fit(env, nb_steps=10000, visualize=False, verbose=1)






# # load image
# img = cv.imread(link, cv.IMREAD_GRAYSCALE)

# #OCR on initial image
# ocr_results = ocr(img,langs,gpu)
# # loop over the words
# for (bbox, text, prob) in ocr_results:

#     # unpack the bounding box
#     (tl, tr, br, bl) = bbox
#     tl = (int(tl[0]), int(tl[1]))
#     tr = (int(tr[0]), int(tr[1]))
#     br = (int(br[0]), int(br[1]))
#     bl = (int(bl[0]), int(bl[1]))

#     cropped_img = img[tl[1]:br[1], tl[0]:br[0]]
#     # cleanup the text and draw the box surrounding the text along with the OCR'd text itself
#     text = cleanup_text(text)

#     # Bounding box for individual characters
#     cropped_img = cv.adaptiveThreshold(cropped_img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 21,5)
#     cropped_img = pad(cv.resize(cropped_img,(cropped_img.shape[1]*10,cropped_img.shape[0]*10),interpolation = cv.INTER_LINEAR),330,3300)

# # show the output image
# plt.figure(figsize=(15,15))
# plt.imshow(cropped_img)
# plt.show()
