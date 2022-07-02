import cv2 as cv
import matplotlib.pyplot as plt
from Helper import cleanup_text,preprocessing,postprocess,pad,ocr
link = 'Resources/sample2.png'
langs = ['en']
gpu = False
in_shape = [71,220,1]

# load image
img = cv.imread(link, cv.IMREAD_GRAYSCALE)

#OCR on initial image
ocr_results = ocr(img,langs,gpu)
# loop over the words
for (bbox, text, prob) in ocr_results:

	# unpack the bounding box
	(tl, tr, br, bl) = bbox
	tl = (int(tl[0]), int(tl[1]))
	tr = (int(tr[0]), int(tr[1]))
	br = (int(br[0]), int(br[1]))
	bl = (int(bl[0]), int(bl[1]))

	cropped_img = img[tl[1]:br[1], tl[0]:br[0]]
	# cleanup the text and draw the box surrounding the text along with the OCR'd text itself
	text = cleanup_text(text)
 
	# Bounding box for individual characters
	print(f"Image size: {cropped_img.shape}")
	cropped_img = pad(cv.resize(cropped_img,(cropped_img.shape[1],cropped_img.shape[0]),interpolation = cv.INTER_LINEAR),in_shape[0],in_shape[1])
	
# # show the output image
# plt.figure(figsize=(15,15))
# plt.imshow(cropped_C_img)
# plt.show()
