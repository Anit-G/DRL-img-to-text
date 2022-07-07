import matplotlib.pyplot as plt
from Helper import *
from models import *
from gym import Env
from gym.spaces import Discrete, Box
import random


link = 'Resources/sample3.png'
langs = ['en']
gpu = True
in_shape = [72,250,3]
Train = True

class pdfread(Env):
	def __init__(self,link):
		log.info('Initialize Enviornment')
		self.action_space = Discrete(26)
		self.observation_space = Box(low=0, high=255, shape=in_shape)

		# start state
		self.state = np.ones(in_shape, dtype=np.uint8)
		self.img = cv.imread(link,cv.IMREAD_GRAYSCALE)
		# self.img = cv.imread(link)
		self.results = ocr(self.img,['en'],False)
		
		self.windex = 0
		self.cindex = 0
		self.c_results = 0
		self.all_test_actions = []
		self.current_action = 0
		# draw the bbox and save image
		bbox_img = drawbbox(self.img,self.results)
		plt.imsave('box_img.png',bbox_img)
		

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
			scale = min(in_shape[0]/cropped_img.shape[0],in_shape[1]/cropped_img.shape[1])
			h = int(cropped_img.shape[0]*scale)
			w = int(cropped_img.shape[1]*scale)
			dim = (w,h)
			# Bounding box for individual characters
			cropped_img = pad(cv.resize(cropped_img,dim,interpolation = cv.INTER_LINEAR),in_shape[0],in_shape[1])
		
			return cropped_img,bbox,text


		# check whether the episode is finished
		if len(self.results)-1 == self.windex:
			done = True
		else:
			done = False

		#saving action space:
		if not Train:
			self.all_test_actions.append(action)
			self.current_action = action

		w_img,wbbox,text = characterR(self.results,self.windex,self.img)	

		# define the state that is going to the model
		input_img = np.zeros(in_shape)
		input_img[:,:,0] = w_img
		input_img[:,:,1] = w_img
		input_img[:,:,2] = w_img

		# input_img = np.expand_dims(input_img,axis=2)
		# print(f"State shape:{input_img.shape} ")
		self.state = input_img
		
        #  reward based on the previous action
		log.info('Logging State information:')
		log.info(f'Current action: {action}, Current word: {text}, current character: {self.cindex}')
		reward = get_reward(action,text,self.cindex)

		#update index
		if self.cindex == len(text)-1:
			log.info(f'[INFO] Moving to next word: Current word: {text}, (Current index): {self.windex}')
			self.cindex = 0
			self.windex+=1
			cv.imwrite(f"words/{text}.png",w_img)
		else:	
			#update character index
			self.cindex +=1
		
		# placeholder used for diagnosis may contain probabilites
		info = {}
		
		return self.state, reward, done, info

	def render(self):
		# i don't need to render any visual
		# display action, current state (pic + text) and reward
		print(self.current_action)
		with open('render.txt','w') as f:
			f.write(f'Current action: {self.current_action} Current index: {self.windex}')
		pass

	def reset(self):
		# reset states
		self.state = np.ones(in_shape, dtype=np.uint8)
		self.img = cv.imread(link,cv.IMREAD_GRAYSCALE)
		self.results = ocr(self.img,['en'],False)
		self.windex = 0
		self.cindex = 0
		self.c_results = 0
		return self.state


if __name__ == "__main__":

	env = pdfread(link=link)
	states = env.observation_space.shape
	actions = env.action_space.n

	print(f"Size of states: {states}")
	print(f"Number of actions: {actions}")

	model = LSTM_model(states, actions)
	model.summary()

	dqn = build_agent(model, actions)
	dqn.compile(Adam(lr=1e-3), metrics=['mae'])

	dqn.fit(env, nb_steps=30000, visualize=False, verbose=1)
	dqn.save_weights('Weights/dqn_weights1.h5f',overwrite=True)

	# if Train:
	# 	# train transfer model
	# 	dqn.fit(env, nb_steps=30000, visualize=False, verbose=1)
	# 	dqn.save_weights('Weights/dqn_weights1.h5f',overwrite=True)
	# 	scores = dqn.test(env,nb_episodes=1,visualize=False)
	# 	print(np.mean(scores.history['episode_reward']))
	# 	# _ = dqn.test(env, nb_episodes=15, visualize=True)
	# else:
	# 	# load weights
	# 	print('Loading Weights')
	# 	dqn.load_weights('Weights/dqn_weights2.h5f') 
	# 	scores = dqn.test(env,nb_episodes=10,visualize=False)
	# 	print(np.mean(scores.history['episode_reward']))
	# 	print(env.all_test_actions)