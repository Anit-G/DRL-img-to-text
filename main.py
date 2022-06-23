import matplotlib.pyplot as plt
from Helper import *
from gym import Env
from gym.spaces import Discrete, Box
import random
link = 'Resources/sample.png'
langs = ['en']
gpu = False


class pdfread(Env):
	def __init__(self,link,size):
		self.action_space = Discrete(26)
		self.observation_space = Box(low=0, high=255, shape=(size,size,1,))

		# start state
		self.state = np.ones([size,size,1], dtype=np.uint8)
		self.img = cv.imread(link,cv.IMREAD_GRAYSCALE)
		# self.img = cv.imread(link)
		self.results = ocr(self.img,['en'],False)
		self.windex = 0
		self.cindex = 0
		self.c_results = 0
		

	def step(self,action):
		def characterR(result,w_index,c_index,img):
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

			cropped_img = img[tl[1]:br[1], tl[0]:br[0]]
			# cleanup the text in the word bounding box
			text = cleanup_text(text)

			# Bounding box for individual characters
			cropped_img = cv.resize(cropped_img,(cropped_img.shape[1]*10,cropped_img.shape[0]*10),interpolation = cv.INTER_LINEAR)
			C_img,characterbbox = preprocessing(cropped_img,31,1,600.0)

			#sort character bbox
			characterbbox = postprocess(characterbbox)
			if len(characterbbox)<1:
				print('\nWe got a problem')
				print(f"number of boxs: {len(characterbbox)}")
				print(f"text: {text}")
				print(f'index: {c_index}')
				print(C_img.shape)
				plt.imsave(f'stuff{w_index}.png',cropped_img)
				plt.imsave(f'stuff2{w_index}.png',C_img)

				# initiate an empty character box
				shape = C_img.shape
				characterbbox = []
				characterbbox.append([1,1,int(shape[1]-5),int(shape[0]-5)])

			#use character tracker
			cbbox = characterbbox[c_index]
			cropped_C_img = C_img[cbbox[1]:int(cbbox[1]+cbbox[3]),cbbox[0]:int(cbbox[0]+cbbox[2])]

			# pad the image into a square of size 150,150
			cropped_C_img = pad(cropped_C_img)

			# check if the number of characters in the word align with the number of the bounding box
			if len(characterbbox)!=len(text):
				print(f"The word --{text}-- does not have an appropreiate amount of bounding boxes")
				# print(characterbbox)
				# plt.figure(figsize=(15,15))
				# plt.imshow(drawbbox(cropped_C_img,characterbbox))
				# plt.show()
			
			return cropped_C_img,cbbox,text
		# check whether the episode is finished
		if len(self.results)-1 == self.windex:
			done = True
		else:
			done = False

		# check if the we are currently on the zeroth character of a word
		# the function only needs to initiate at the start of a new word
		if self.cindex == 0:
			self.c_results= characterR(self.results,self.windex,self.cindex,self.img)	

		c_img,cbbox,text = self.c_results

		# define the state that is going to the model
		self.state = np.expand_dims(c_img,axis=2)
		
		# print(f"image shape: {c_img.shape}")
		# print(f"state shape: {self.state.shape}")
		# print(text)
		# print(action)
		# print(self.cindex)
		# print(len(text))

		# reward based on the previous action
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
		self.state = np.ones([200,200,1], dtype=np.uint8)
		#self.img = cv.imread(link,cv.IMREAD_GRAYSCALE)
		#self.results = ocr(self.img,['en'],False)
		self.windex = 0
		self.cindex = 0
		self.c_results = 0
		return self.state

env = pdfread(link=link,size=200)
states = env.observation_space.shape
actions = env.action_space.n

print(f"Size of states: {states}")
print(f"Number of actions: {actions}")

# episodes = 1
# for episode in range(1, episodes+1):
#     state = env.reset()
#     done = False
#     score = 0 
#     while not done:
#         env.render()
#         action = random.choice([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25])
#         n_state, reward, done, info = env.step(action)
#         score+=reward
#     print('Episode:{} Score:{}'.format(episode, score))
# env.close()
# del model
model = build_model(states,actions)
model.summary()

dqn = build_agent(model, actions)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])
dqn.fit(env, nb_steps=50000, visualize=False, verbose=1)


# # load image
# img = cv.imread(link, cv.IMREAD_GRAYSCALE)

# #OCR on initial image
# ocr_results = ocr(img,langs,gpu)
# # loop over the words
# for (bbox, text, prob) in ocr_results:

# 	# unpack the bounding box
# 	(tl, tr, br, bl) = bbox
# 	tl = (int(tl[0]), int(tl[1]))
# 	tr = (int(tr[0]), int(tr[1]))
# 	br = (int(br[0]), int(br[1]))
# 	bl = (int(bl[0]), int(bl[1]))

# 	cropped_img = img[tl[1]:br[1], tl[0]:br[0]]
# 	# cleanup the text and draw the box surrounding the text along with the OCR'd text itself
# 	text = cleanup_text(text)
 
# 	# Bounding box for individual characters
# 	cropped_img = cv.resize(cropped_img,(cropped_img.shape[1]*10,cropped_img.shape[0]*10),interpolation = cv.INTER_LINEAR)
# 	C_img,characterbbox = preprocessing(cropped_img,31,1,600.0)
# 	#sort character bbox
# 	characterbbox = postprocess(characterbbox)
# 	#C_img = drawbbox(C_img,characterbbox)
# 	for i, cbbox in enumerate(characterbbox):
# 		cropped_C_img = C_img[cbbox[1]:int(cbbox[1]+cbbox[3]),cbbox[0]:int(cbbox[0]+cbbox[2])]
	
# 		# pad the image into a square of size 150,150
# 		cropped_C_img = pad(cropped_C_img)
# 		# print(characterbbox)
# 		# plt.figure(figsize=(15,15))
	
# 		plt.imsave(f'image_{text}_{i}.png',cropped_C_img)

# # show the output image
# plt.figure(figsize=(15,15))
# plt.imshow(cropped_C_img)
# plt.show()
