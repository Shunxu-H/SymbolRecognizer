import cv2
import trainModel 
import tensorflow as tf
import numpy as np
import Utility

from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets

drawing = False # true if mouse is pressed
width = 280;
height = 280;
target_width = 28;
target_height = 28;
xRatio = int(width/target_width);
yRatio = int(height/target_height);
sessionName = 'my-model'

# restore tensorflow session
sess = tf.Session()
sess.run(tf.initialize_all_variables())
saver = tf.train.import_meta_graph('my-model.meta')
saver.restore(sess, tf.train.latest_checkpoint('./'))

all_vars = tf.get_collection('W')
for v in all_vars:
    v_ = sess.run(v)

all_vars = tf.get_collection('B')
for v in all_vars:
    v_ = sess.run(v)

# mouse callback function
def draw_circle(event,x,y,flags,param):
    global drawing

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            cv2.circle(img,(x,y),7,(255,255,255),-1)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.circle(img,(x,y),7,(255,255,255),-1)

img = np.full(shape=(width,height,3), fill_value=0, dtype=np.uint8)
cv2.namedWindow('digitRecognizer')
cv2.setMouseCallback('digitRecognizer',draw_circle)

cv2.moveWindow('digitRecognizer', 100, 100) 


while(1):
    cv2.imshow('digitRecognizer',img)
    k = cv2.waitKey(1) & 0xFF
    if k == 13:
    	# cv2.namedWindow('compressedImg');
    	compressedImg = np.zeros(shape=(target_width,target_height,1), dtype=np.float32);
    	x = 0;
    	y = 0;
    	print(img.shape[0]);
    	while (x < img.shape[0]):
    		while(y < img.shape[1]):

    			# print(str(img.shape));
    			# print(str(np.amin(img[x:x+xRatio, y:y+yRatio])));
    			# compressedImg[int(x/xRatio), int(y/xRatio)] = np.amax(img[x:x+xRatio, y:y+yRatio]);
    			compressedImg[int(y/xRatio), int(x/xRatio), 0] = np.average(img[y:y+yRatio, x:x+xRatio])/255.0;	
    			y += yRatio;
    		y = 0;
    		x += xRatio;


    	fd = {tf.get_collection('X')[0]: [compressedImg]}
    	result = sess.run(tf.get_collection('Y')[0], fd);
    	l = list(result[0])

    	print("Prediction: " + str(l.index(max(l))));

    	# print(result);

    	# mnist = read_data_sets("data", one_hot=True, reshape=False, validation_size=0)
    	# batch_X, batch_Y = mnist.train.next_batch(100)


    	# result = sess.run(tf.get_collection('Y')[0], feed_dict={tf.get_collection('X')[0]: [batch_X[0]]});


    	# print(result)
    	# print(batch_Y[0])
    	# Utility.displayImg(compressedImg, width, height, target_width, target_height)


    	# fd = {tf.get_collection('X')[0]: [batch_X[0]]}
    	# result = sess.run(tf.get_collection('Y')[0], fd);
    	# l = list(result[0])

    	# print(l.index(max(l)));

    	# print(result);

    	img = np.full(shape=(width,height,3), fill_value=0, dtype=np.uint8)
        # print("Enter hit")
    elif k == 27:
        break

cv2.destroyAllWindows()
