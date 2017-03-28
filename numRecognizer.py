import cv2
import trainModel 
import tensorflow as tf
import numpy as np

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
            cv2.circle(img,(x,y),3,(100,100,100),-1)
            cv2.circle(img,(x,y),1,(250,250,250),-1)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.circle(img,(x,y),3,(100,100,100),-1)
        cv2.circle(img,(x,y),1,(250,250,250),-1)

img = np.full(shape=(width,height,3), fill_value=0, dtype=np.uint8)
cv2.namedWindow('digitRecognizer')
cv2.setMouseCallback('digitRecognizer',draw_circle)


while(1):
    cv2.imshow('digitRecognizer',img)
    k = cv2.waitKey(1) & 0xFF
    if k == 13:
    	cv2.namedWindow('compressedImg');
    	compressedImg = np.zeros(shape=(target_width,target_height,1), dtype=np.float32);
    	x = 0;
    	y = 0;
    	print(img.shape[0]);
    	while (x < img.shape[0]):
    		while(y < img.shape[1]):

    			# print(str(img.shape));
    			# print(str(np.amin(img[x:x+xRatio, y:y+yRatio])));
    			# compressedImg[int(x/xRatio), int(y/xRatio)] = np.amax(img[x:x+xRatio, y:y+yRatio]);
    			compressedImg[int(y/xRatio), int(x/xRatio), 0] = np.average(img[x:x+xRatio, y:y+yRatio])/255.0;	
    			y += yRatio;
    		y = 0;
    		x += xRatio;
    		# print(str(x) + " " + str(y) + " " + str(x < img.shape[0]));

    	# data = list(compressedImg)
    	# adata = [float(x) for x in data]
    	# sess.run(tf.global_variables_initializer())

    	# X = tf.placeholder(tf.float32, [None, target_width, target_height, 1])
    	# # Y = tf.nn.softmax(tf.matmul(compressedImg, tf.get_collection('W')[4]) + tf.get_collection('B')[4])
    	# W1 = tf.get_collection('W')[0]
    	# B1 = tf.get_collection('B')[0]
    	# W2 = tf.get_collection('W')[1]
    	# B2 = tf.get_collection('B')[1]
    	# W3 = tf.get_collection('W')[2]
    	# B3 = tf.get_collection('B')[2]
    	# W4 = tf.get_collection('W')[3]
    	# B4 = tf.get_collection('B')[3]
    	# W5 = tf.get_collection('W')[4]
    	# B5 = tf.get_collection('B')[4]

    	# # XX = tf.reshape(adata, [-1, target_width*target_height])
    	# X = tf.placeholder(tf.float32, [None, target_width*target_height])
    	# Y1 = tf.nn.relu(tf.matmul(X, W1) + B1)
    	# Y2 = tf.nn.relu(tf.matmul(Y1, W2) + B2)
    	# Y3 = tf.nn.relu(tf.matmul(Y2, W3) + B3)
    	# Y4 = tf.nn.relu(tf.matmul(Y3, W4) + B4)
    	# Ylogits = tf.matmul(Y4, W5) + B5
    	# Y  = tf.nn.softmax(Ylogits)
    	print(compressedImg)
    	fd = {tf.get_collection('X')[0]: [compressedImg]}
    	result = sess.run(tf.get_collection('Y')[0], fd);
    	l = list(result[0])

    	print(l.index(max(l)));

    	print(result);

    	mnist = read_data_sets("data", one_hot=True, reshape=False, validation_size=0)
    	batch_X, batch_Y = mnist.train.next_batch(100)


    	# result = sess.run(tf.get_collection('Y')[0], feed_dict={tf.get_collection('X')[0]: [batch_X[0]]});


    	# print(result)
    	# print(batch_Y[0])
    	flatColor = batch_X[0];
    	peek = np.zeros(shape=(width, height, 3), dtype=np.float32);
    	i = 0;
    	for x in range(target_width):
    		for y in range(target_height):
		    	for i in range(10):
		    		for j in range(10):
		    			peek[x*10+i, y*10+j, 0] = flatColor[x, y, 0];
		    			peek[x*10+i, y*10+j, 1] = flatColor[x, y, 0];
		    			peek[x*10+i, y*10+j, 2] = flatColor[x, y, 0];


    	cv2.imshow('compressedImg',peek)
    	cv2.moveWindow('compressedImg', 100, 100) 
    	cv2.moveWindow('digitRecognizer', 100, 100) 

    	img = np.full(shape=(width,height,3), fill_value=0, dtype=np.uint8)
        # print("Enter hit")
    elif k == 27:
        break

cv2.destroyAllWindows()