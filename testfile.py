import cv2 as cv
from Helper import preprocessing, postprocess
import matplotlib.pyplot as plt
link = 'stuff.png'
img = cv.imread(link,cv.IMREAD_GRAYSCALE)
img2 = cv.imread('stuff2.png',cv.IMREAD_GRAYSCALE)
# Bounding box for individual characters
cropped_img = img
cropped_img = cv.resize(cropped_img,(cropped_img.shape[1]*10,cropped_img.shape[0]*10),interpolation = cv.INTER_LINEAR)
C_img,characterbbox = preprocessing(img2,31,1,600.0)
#sort character bbox
# characterbbox = postprocess(characterbbox)

# plt.figure()
# plt.imshow(cropped_img)
# plt.show()

plt.figure()
plt.imshow(C_img)
print(characterbbox)
plt.show()