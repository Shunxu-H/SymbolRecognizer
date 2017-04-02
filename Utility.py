import cv2
import numpy as np

def displayImg(img, width, height, target_width, target_height):
	peek = np.zeros(shape=(width, height, 3), dtype=np.float32);
	x = y = i = j = 0;
	for x in range(target_width):
		for y in range(target_height):
			for i in range(10):
				for j in range(10):
					peek[x*10+i, y*10+j, 0] = img[x, y, 0];
					peek[x*10+i, y*10+j, 1] = img[x, y, 0];
					peek[x*10+i, y*10+j, 2] = img[x, y, 0];
	cv2.namedWindow('compressedImg');
	cv2.imshow('compressedImg',peek);