import cv2
import os

imgs = os.listdir('./')
print(imgs)

for img in imgs:
	if img.endswith('.png'):
		im = cv2.imread(img)
		im = cv2.resize(im, (256, 256))
		cv2.imwrite(img, im)