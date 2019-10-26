import os
import tensorflow as tf

if __name__=="__main__":
#	os.environ['CUDA_VISIBLE_DEVICES']='0'

	if tf.test.gpu_device_name():
		print('GPU found')
	else:
		print("No GPU found")
