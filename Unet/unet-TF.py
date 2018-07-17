# -*- coding:utf-8 -*-
'''
#====#====#====#====
# Project Name:     U-net
# File Name:        unet-TF
# Date:             2/10/18 2:33 PM
# Using IDE:        PyCharm Community Edition
# From HomePage:    https://github.com/DuFanXin/U-net
# Author:           DuFanXin
# BlogPage:         http://blog.csdn.net/qq_30239975
# E-mail:           18672969179@163.com
# Copyright (c) 2018, All Rights Reserved.
#====#====#====#====
'''
import tensorflow as tf
import argparse
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"

TRAIN_SET_NAME = 'train_set.tfrecords'
VALIDATION_SET_NAME = 'validation_set.tfrecords'
TEST_SET_NAME = 'test_set.tfrecords'
TFRECORD_PATH = '../data_set/my_set'
ORIGIN_PREDICT_DIRECTORY = '../data_set/test'
MODEL_DIR = '../data_set/model/model.ckpt'
LOG_DIR = '../data_set/logs'
INPUT_IMG_WIDE, INPUT_IMG_HEIGHT, INPUT_IMG_CHANNEL = 512, 512, 1
OUTPUT_IMG_WIDE, OUTPUT_IMG_HEIGHT, OUTPUT_IMG_CHANNEL = 512, 512, 1
TRAIN_SET_SIZE = 8
VALIDATION_SET_SIZE = 27
TEST_SET_SIZE = 30
PREDICT_SET_SIZE = 30
EPOCH_NUM = 1
TRAIN_BATCH_SIZE = 1
VALIDATION_BATCH_SIZE = 1
TEST_BATCH_SIZE = 1
PREDICT_BATCH_SIZE = 1
PREDICT_SAVED_DIRECTORY = '../data_set/predictions'
EPS = 10e-5
CLASS_NUM = 2


def calculate_unet_input_and_output(bottom=0):
	# 从最底层右边开始计算网络的输入输出的图片大小
	y, z = bottom + 2, bottom * 2
	y, z = y + 2, z - 2
	y, z = y * 2, z - 2
	y, z = y + 2, z * 2
	y, z = y + 2, z - 2
	y, z = y * 2, z - 2
	y, z = y + 2, z * 2
	y, z = y + 2, z - 2
	y, z = y * 2, z - 2
	y, z = y + 2, z * 2
	y, z = y + 2, z - 2
	y, z = y * 2, z - 2
	y += 4
	print(y)
	print(z)





def read_check_tfrecords():
	import cv2
	train_file_path = os.path.join(TFRECORD_PATH, TRAIN_SET_NAME)
	#string_input_producer第一个参数为列表
	train_image_filename_queue = tf.train.string_input_producer(string_tensor=[train_file_path], num_epochs=1, shuffle=True)
	train_images, train_labels = read_image(train_image_filename_queue)#从tfrecord文件中读取一个图片显示
	with tf.Session() as sess:  # 开始一个会话,用来输出图像
		sess.run(tf.global_variables_initializer())
		sess.run(tf.local_variables_initializer())
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(coord=coord)
		images,labels = sess.run([train_images,train_labels])
		cv2.imshow('image', images)
		cv2.imshow('lael', labels * 255)#label中0，1的像素
		cv2.waitKey(0)
		coord.request_stop()
		coord.join(threads)
	print("Done reading and checking")


def read_image(file_queue):
	reader = tf.TFRecordReader()
	# key, value = reader.read(file_queue)
	_, serialized_example = reader.read(file_queue)
	features = tf.parse_single_example(
		serialized_example,
		features={
			'label': tf.FixedLenFeature([], tf.string),
			'image_raw': tf.FixedLenFeature([], tf.string)
			})

	image = tf.decode_raw(features['image_raw'], tf.uint8)
	# print('image ' + str(image))
	image = tf.reshape(image, [INPUT_IMG_WIDE, INPUT_IMG_HEIGHT, INPUT_IMG_CHANNEL])
	# image = tf.image.convert_image_dtype(image, dtype=tf.float32)
	# image = tf.image.resize_images(image, (IMG_HEIGHT, IMG_WIDE))
	# image = tf.cast(image, tf.float32) * (1. / 255) - 0.5

	label = tf.decode_raw(features['label'], tf.uint8)
	# label = tf.cast(label, tf.int64)
	label = tf.reshape(label, [OUTPUT_IMG_WIDE, OUTPUT_IMG_HEIGHT])
	# label = tf.decode_raw(features['image_raw'], tf.uint8)
	# print(label)
	# label = tf.reshape(label, shape=[1, 4])

	return image, label


def read_image_batch(file_queue, batch_size):
	img, label = read_image(file_queue)
	min_after_dequeue = 2000
	capacity = 4000
	# image_batch, label_batch = tf.train.batch([img, label], batch_size=batch_size, capacity=capacity, num_threads=10)
	image_batch, label_batch = tf.train.shuffle_batch(
		tensors=[img, label], batch_size=batch_size,
		capacity=capacity, min_after_dequeue=min_after_dequeue)
	# one_hot_labels = tf.to_float(tf.one_hot(indices=label_batch, depth=CLASS_NUM))
	one_hot_labels = tf.reshape(label_batch, [batch_size, OUTPUT_IMG_HEIGHT, OUTPUT_IMG_WIDE])
	return image_batch, one_hot_labels


class Unet:

	def __init__(self):
		print('New U-net Network')
		self.input_image = None
		self.input_label = None
		self.cast_image = None
		self.cast_label = None
		self.keep_prob = None
		self.lamb = None
		self.result_expand = None
		self.loss, self.loss_mean, self.loss_all, self.train_step = [None] * 4
		self.prediction, self.correct_prediction, self.accuracy = [None] * 3
		self.result_conv = {}
		self.result_relu = {}
		self.result_maxpool = {}
		self.result_from_contract_layer = {}
		self.w = {}
		self.b = {}

	def init_w(self, shape, name):
		with tf.name_scope('init_w'):
			stddev = tf.sqrt(x=2 / (shape[0] * shape[1] * shape[2]))
			# stddev = 0.01
			w = tf.Variable(initial_value=tf.truncated_normal(shape=shape, stddev=stddev, dtype=tf.float32), name=name)
			tf.add_to_collection(name='loss', value=tf.contrib.layers.l2_regularizer(self.lamb)(w))
			return w

	@staticmethod
	def init_b(shape, name):
		with tf.name_scope('init_b'):
			return tf.Variable(initial_value=tf.random_normal(shape=shape, dtype=tf.float32), name=name)

	@staticmethod
	def copy_and_crop_and_merge(result_from_contract_layer, result_from_upsampling):
		# result_from_contract_layer_shape = tf.shape(result_from_contract_layer)
		# result_from_upsampling_shape = tf.shape(result_from_upsampling)
		# result_from_contract_layer_crop = \
		# 	tf.slice(
		# 		input_=result_from_contract_layer,
		# 		begin=[
		# 			0,
		# 			(result_from_contract_layer_shape[1] - result_from_upsampling_shape[1]) // 2,
		# 			(result_from_contract_layer_shape[2] - result_from_upsampling_shape[2]) // 2,
		# 			0
		# 		],
		# 		size=[
		# 			result_from_upsampling_shape[0],
		# 			result_from_upsampling_shape[1],
		# 			result_from_upsampling_shape[2],
		# 			result_from_upsampling_shape[3]
		# 		]
		# 	)
		result_from_contract_layer_crop = result_from_contract_layer
		return tf.concat(values=[result_from_contract_layer_crop, result_from_upsampling], axis=-1)

	def set_up_unet(self, batch_size):
		# input
		with tf.name_scope('input'):
			# learning_rate = tf.train.exponential_decay()
			self.input_image = tf.placeholder(
				dtype=tf.float32, shape=[batch_size, INPUT_IMG_WIDE, INPUT_IMG_WIDE, INPUT_IMG_CHANNEL], name='input_images'
			)
			# self.cast_image = tf.reshape(
			# 	tensor=self.input_image,
			# 	shape=[batch_size, INPUT_IMG_WIDE, INPUT_IMG_WIDE, INPUT_IMG_CHANNEL]
			# )

			# for softmax_cross_entropy_with_logits(labels=self.input_label, logits=self.prediction, name='loss')
			# using one-hot
			# self.input_label = tf.placeholder(
			# 	dtype=tf.uint8, shape=[OUTPUT_IMG_WIDE, OUTPUT_IMG_WIDE], name='input_labels'
			# )
			# self.cast_label = tf.reshape(
			# 	tensor=self.input_label,
			# 	shape=[batch_size, OUTPUT_IMG_WIDE, OUTPUT_IMG_HEIGHT]
			# )

			# for sparse_softmax_cross_entropy_with_logits(labels=self.input_label, logits=self.prediction, name='loss')
			# not using one-hot coding
			self.input_label = tf.placeholder(
				dtype=tf.int32, shape=[batch_size, OUTPUT_IMG_WIDE, OUTPUT_IMG_WIDE], name='input_labels'
			)
			self.keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob')
			self.lamb = tf.placeholder(dtype=tf.float32, name='lambda')

		# layer1两个卷积加一个池化，1-5都是这样， 输入:515*512,输出256*256
		with tf.name_scope('layer_1'):
			# conv_1
			self.w[1] = self.init_w(shape=[3, 3, INPUT_IMG_CHANNEL, 64], name='w_1')
			self.b[1] = self.init_b(shape=[64], name='b_1')
			result_conv_1 = tf.nn.conv2d(
				input=self.input_image, filter=self.w[1],
				strides=[1, 1, 1, 1], padding='SAME', name='conv_1')
			result_relu_1 = tf.nn.relu(tf.nn.bias_add(result_conv_1, self.b[1], name='add_bias'), name='relu_1')

			# conv_2
			self.w[2] = self.init_w(shape=[3, 3, 64, 64], name='w_2')
			self.b[2] = self.init_b(shape=[64], name='b_2')
			result_conv_2 = tf.nn.conv2d(
				input=result_relu_1, filter=self.w[2],
				strides=[1, 1, 1, 1], padding='SAME', name='conv_2')
			result_relu_2 = tf.nn.relu(tf.nn.bias_add(result_conv_2, self.b[2], name='add_bias'), name='relu_2')
			self.result_from_contract_layer[1] = result_relu_2  # 该层结果临时保存, 供上采样使用

			# maxpool
			result_maxpool = tf.nn.max_pool(
				value=result_relu_2, ksize=[1, 2, 2, 1],
				strides=[1, 2, 2, 1], padding='VALID', name='maxpool')

			# dropout
			result_dropout = tf.nn.dropout(x=result_maxpool, keep_prob=self.keep_prob)

		# layer 2，输入：256*256，输出128*128
		with tf.name_scope('layer_2'):
			# conv_1
			self.w[3] = self.init_w(shape=[3, 3, 64, 128], name='w_3')
			self.b[3] = self.init_b(shape=[128], name='b_3')
			result_conv_1 = tf.nn.conv2d(
				input=result_dropout, filter=self.w[3],
				strides=[1, 1, 1, 1], padding='SAME', name='conv_1')
			result_relu_1 = tf.nn.relu(tf.nn.bias_add(result_conv_1, self.b[3], name='add_bias'), name='relu_1')

			# conv_2
			self.w[4] = self.init_w(shape=[3, 3, 128, 128], name='w_4')
			self.b[4] = self.init_b(shape=[128], name='b_4')
			result_conv_2 = tf.nn.conv2d(
				input=result_relu_1, filter=self.w[4],
				strides=[1, 1, 1, 1], padding='SAME', name='conv_2')
			result_relu_2 = tf.nn.relu(tf.nn.bias_add(result_conv_2, self.b[4], name='add_bias'), name='relu_2')
			self.result_from_contract_layer[2] = result_relu_2  # 该层结果临时保存, 供上采样使用

			# maxpool
			result_maxpool = tf.nn.max_pool(
				value=result_relu_2, ksize=[1, 2, 2, 1],
				strides=[1, 2, 2, 1], padding='VALID', name='maxpool')

			# dropout
			result_dropout = tf.nn.dropout(x=result_maxpool, keep_prob=self.keep_prob)

		# layer 3，输入：128*128，输出：64*64
		with tf.name_scope('layer_3'):
			# conv_1
			self.w[5] = self.init_w(shape=[3, 3, 128, 256], name='w_5')
			self.b[5] = self.init_b(shape=[256], name='b_5')
			result_conv_1 = tf.nn.conv2d(
				input=result_dropout, filter=self.w[5],
				strides=[1, 1, 1, 1], padding='SAME', name='conv_1')
			result_relu_1 = tf.nn.relu(tf.nn.bias_add(result_conv_1, self.b[5], name='add_bias'), name='relu_1')

			# conv_2
			self.w[6] = self.init_w(shape=[3, 3, 256, 256], name='w_6')
			self.b[6] = self.init_b(shape=[256], name='b_6')
			result_conv_2 = tf.nn.conv2d(
				input=result_relu_1, filter=self.w[6],
				strides=[1, 1, 1, 1], padding='SAME', name='conv_2')
			result_relu_2 = tf.nn.relu(tf.nn.bias_add(result_conv_2, self.b[6], name='add_bias'), name='relu_2')
			self.result_from_contract_layer[3] = result_relu_2  # 该层结果临时保存, 供上采样使用

			# maxpool
			result_maxpool = tf.nn.max_pool(
				value=result_relu_2, ksize=[1, 2, 2, 1],
				strides=[1, 2, 2, 1], padding='VALID', name='maxpool')

			# dropout
			result_dropout = tf.nn.dropout(x=result_maxpool, keep_prob=self.keep_prob)

		# layer 4，输入64*64，输出32*32
		with tf.name_scope('layer_4'):
			# conv_1
			self.w[7] = self.init_w(shape=[3, 3, 256, 512], name='w_7')
			self.b[7] = self.init_b(shape=[512], name='b_7')
			result_conv_1 = tf.nn.conv2d(
				input=result_dropout, filter=self.w[7],
				strides=[1, 1, 1, 1], padding='SAME', name='conv_1')
			result_relu_1 = tf.nn.relu(tf.nn.bias_add(result_conv_1, self.b[7], name='add_bias'), name='relu_1')

			# conv_2
			self.w[8] = self.init_w(shape=[3, 3, 512, 512], name='w_8')
			self.b[8] = self.init_b(shape=[512], name='b_8')
			result_conv_2 = tf.nn.conv2d(
				input=result_relu_1, filter=self.w[8],
				strides=[1, 1, 1, 1], padding='SAME', name='conv_2')
			result_relu_2 = tf.nn.relu(tf.nn.bias_add(result_conv_2, self.b[8], name='add_bias'), name='relu_2')
			self.result_from_contract_layer[4] = result_relu_2  # 该层结果临时保存, 供上采样使用
			print("layer_4:{result_conv_2:%s}"%(str(result_relu_2.shape)))
			# maxpool
			result_maxpool = tf.nn.max_pool(
				value=result_relu_2, ksize=[1, 2, 2, 1],
				strides=[1, 2, 2, 1], padding='VALID', name='maxpool')

			# dropout
			result_dropout = tf.nn.dropout(x=result_maxpool, keep_prob=self.keep_prob)

		# layer 5 (bottom)，两个卷积加一个反卷积，输入32*32，输出64*64
		with tf.name_scope('layer_5'):
			# conv_1
			self.w[9] = self.init_w(shape=[3, 3, 512, 1024], name='w_9')
			self.b[9] = self.init_b(shape=[1024], name='b_9')
			result_conv_1 = tf.nn.conv2d(
				input=result_dropout, filter=self.w[9],
				strides=[1, 1, 1, 1], padding='SAME', name='conv_1')
			result_relu_1 = tf.nn.relu(tf.nn.bias_add(result_conv_1, self.b[9], name='add_bias'), name='relu_1')

			# conv_2
			self.w[10] = self.init_w(shape=[3, 3, 1024, 1024], name='w_10')
			self.b[10] = self.init_b(shape=[1024], name='b_10')
			result_conv_2 = tf.nn.conv2d(
				input=result_relu_1, filter=self.w[10],
				strides=[1, 1, 1, 1], padding='SAME', name='conv_2')
			result_relu_2 = tf.nn.relu(tf.nn.bias_add(result_conv_2, self.b[10], name='add_bias'), name='relu_2')

			# up sample
			self.w[11] = self.init_w(shape=[2, 2, 512, 1024], name='w_11')
			self.b[11] = self.init_b(shape=[512], name='b_11')
			result_up = tf.nn.conv2d_transpose(
				value=result_relu_2, filter=self.w[11],
				output_shape=[batch_size, 64, 64, 512],
				strides=[1, 2, 2, 1], padding='VALID', name='Up_Sample')
			result_relu_3 = tf.nn.relu(tf.nn.bias_add(result_up, self.b[11], name='add_bias'), name='relu_3')

			# dropout
			result_dropout = tf.nn.dropout(x=result_relu_3, keep_prob=self.keep_prob)

		# layer 6：输入64*64，输出128*128
		with tf.name_scope('layer_6'):
			# copy, crop and merge
			result_merge = self.copy_and_crop_and_merge(
				result_from_contract_layer=self.result_from_contract_layer[4], result_from_upsampling=result_dropout)
			print(self.result_from_contract_layer[4])
			print(result_dropout)
			print(result_merge)

			# conv_1
			self.w[12] = self.init_w(shape=[3, 3, 1024, 512], name='w_12')
			self.b[12] = self.init_b(shape=[512], name='b_12')
			result_conv_1 = tf.nn.conv2d(
				input=result_merge, filter=self.w[12],
				strides=[1, 1, 1, 1], padding='SAME', name='conv_1')
			result_relu_1 = tf.nn.relu(tf.nn.bias_add(result_conv_1, self.b[12], name='add_bias'), name='relu_1')

			# conv_2
			self.w[13] = self.init_w(shape=[3, 3, 512, 512], name='w_10')
			self.b[13] = self.init_b(shape=[512], name='b_10')
			result_conv_2 = tf.nn.conv2d(
				input=result_relu_1, filter=self.w[13],
				strides=[1, 1, 1, 1], padding='SAME', name='conv_2')
			result_relu_2 = tf.nn.relu(tf.nn.bias_add(result_conv_2, self.b[13], name='add_bias'), name='relu_2')
			# print(result_relu_2.shape[1])

			# up sample
			self.w[14] = self.init_w(shape=[2, 2, 256, 512], name='w_11')
			self.b[14] = self.init_b(shape=[256], name='b_11')
			result_up = tf.nn.conv2d_transpose(
				value=result_relu_2, filter=self.w[14],
				output_shape=[batch_size, 128, 128, 256],
				strides=[1, 2, 2, 1], padding='VALID', name='Up_Sample')
			result_relu_3 = tf.nn.relu(tf.nn.bias_add(result_up, self.b[14], name='add_bias'), name='relu_3')

			# dropout
			result_dropout = tf.nn.dropout(x=result_relu_3, keep_prob=self.keep_prob)

		# layer 7，输入128*128，输出256*256
		with tf.name_scope('layer_7'):
			# copy, crop and merge
			result_merge = self.copy_and_crop_and_merge(
				result_from_contract_layer=self.result_from_contract_layer[3], result_from_upsampling=result_dropout)

			# conv_1
			self.w[15] = self.init_w(shape=[3, 3, 512, 256], name='w_12')
			self.b[15] = self.init_b(shape=[256], name='b_12')
			result_conv_1 = tf.nn.conv2d(
				input=result_merge, filter=self.w[15],
				strides=[1, 1, 1, 1], padding='SAME', name='conv_1')
			result_relu_1 = tf.nn.relu(tf.nn.bias_add(result_conv_1, self.b[15], name='add_bias'), name='relu_1')

			# conv_2
			self.w[16] = self.init_w(shape=[3, 3, 256, 256], name='w_10')
			self.b[16] = self.init_b(shape=[256], name='b_10')
			result_conv_2 = tf.nn.conv2d(
				input=result_relu_1, filter=self.w[16],
				strides=[1, 1, 1, 1], padding='SAME', name='conv_2')
			result_relu_2 = tf.nn.relu(tf.nn.bias_add(result_conv_2, self.b[16], name='add_bias'), name='relu_2')

			# up sample
			self.w[17] = self.init_w(shape=[2, 2, 128, 256], name='w_11')
			self.b[17] = self.init_b(shape=[128], name='b_11')
			result_up = tf.nn.conv2d_transpose(
				value=result_relu_2, filter=self.w[17],
				output_shape=[batch_size, 256, 256, 128],
				strides=[1, 2, 2, 1], padding='VALID', name='Up_Sample')
			result_relu_3 = tf.nn.relu(tf.nn.bias_add(result_up, self.b[17], name='add_bias'), name='relu_3')

			# dropout
			result_dropout = tf.nn.dropout(x=result_relu_3, keep_prob=self.keep_prob)
			print(result_dropout)
		# layer 8，输入256*256，输出512*512
		with tf.name_scope('layer_8'):
			# copy, crop and merge
			result_merge = self.copy_and_crop_and_merge(
				result_from_contract_layer=self.result_from_contract_layer[2], result_from_upsampling=result_dropout)

			# conv_1
			self.w[18] = self.init_w(shape=[3, 3, 256, 128], name='w_12')
			self.b[18] = self.init_b(shape=[128], name='b_12')
			result_conv_1 = tf.nn.conv2d(
				input=result_merge, filter=self.w[18],
				strides=[1, 1, 1, 1], padding='SAME', name='conv_1')
			result_relu_1 = tf.nn.relu(tf.nn.bias_add(result_conv_1, self.b[18], name='add_bias'), name='relu_1')

			# conv_2
			self.w[19] = self.init_w(shape=[3, 3, 128, 128], name='w_10')
			self.b[19] = self.init_b(shape=[128], name='b_10')
			result_conv_2 = tf.nn.conv2d(
				input=result_relu_1, filter=self.w[19],
				strides=[1, 1, 1, 1], padding='SAME', name='conv_2')
			result_relu_2 = tf.nn.relu(tf.nn.bias_add(result_conv_2, self.b[19], name='add_bias'), name='relu_2')

			# up sample
			self.w[20] = self.init_w(shape=[2, 2, 64, 128], name='w_11')
			self.b[20] = self.init_b(shape=[64], name='b_11')
			result_up = tf.nn.conv2d_transpose(
				value=result_relu_2, filter=self.w[20],
				output_shape=[batch_size, 512, 512, 64],
				strides=[1, 2, 2, 1], padding='VALID', name='Up_Sample')
			result_relu_3 = tf.nn.relu(tf.nn.bias_add(result_up, self.b[20], name='add_bias'), name='relu_3')

			# dropout
			result_dropout = tf.nn.dropout(x=result_relu_3, keep_prob=self.keep_prob)
			print(result_dropout)
		# layer 9,三个卷积层作为最后输出
		with tf.name_scope('layer_9'):
			# copy, crop and merge
			result_merge = self.copy_and_crop_and_merge(
				result_from_contract_layer=self.result_from_contract_layer[1], result_from_upsampling=result_dropout)

			# conv_1
			self.w[21] = self.init_w(shape=[3, 3, 128, 64], name='w_12')
			self.b[21] = self.init_b(shape=[64], name='b_12')
			result_conv_1 = tf.nn.conv2d(
				input=result_merge, filter=self.w[21],
				strides=[1, 1, 1, 1], padding='SAME', name='conv_1')
			result_relu_1 = tf.nn.relu(tf.nn.bias_add(result_conv_1, self.b[21], name='add_bias'), name='relu_1')

			# conv_2
			self.w[22] = self.init_w(shape=[3, 3, 64, 64], name='w_10')
			self.b[22] = self.init_b(shape=[64], name='b_10')
			result_conv_2 = tf.nn.conv2d(
				input=result_relu_1, filter=self.w[22],
				strides=[1, 1, 1, 1], padding='SAME', name='conv_2')
			result_relu_2 = tf.nn.relu(tf.nn.bias_add(result_conv_2, self.b[22], name='add_bias'), name='relu_2')

			# convolution to [batch_size, OUTPIT_IMG_WIDE, OUTPUT_IMG_HEIGHT, CLASS_NUM]
			self.w[23] = self.init_w(shape=[1, 1, 64, CLASS_NUM], name='w_11')
			self.b[23] = self.init_b(shape=[CLASS_NUM], name='b_11')
			result_conv_3 = tf.nn.conv2d(
				input=result_relu_2, filter=self.w[23],
				strides=[1, 1, 1, 1], padding='VALID', name='conv_3')
			# self.prediction = tf.nn.relu(tf.nn.bias_add(result_conv_3, self.b[23], name='add_bias'), name='relu_3')
			# self.prediction = tf.nn.sigmoid(x=tf.nn.bias_add(result_conv_3, self.b[23], name='add_bias'), name='sigmoid_1')
			self.prediction = tf.nn.bias_add(result_conv_3, self.b[23], name='add_bias')
		print(self.prediction)
		print(self.input_label)

		# softmax loss
		with tf.name_scope('softmax_loss'):
			# using one-hot
			# self.loss = \
			# 	tf.nn.softmax_cross_entropy_with_logits(labels=self.cast_label, logits=self.prediction, name='loss')

			# not using one-hot
			self.loss = \
				tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_label, logits=self.prediction, name='loss')
			self.loss_mean = tf.reduce_mean(self.loss)
			tf.add_to_collection(name='loss', value=self.loss_mean)
			self.loss_all = tf.add_n(inputs=tf.get_collection(key='loss'))

		# accuracy
		with tf.name_scope('accuracy'):
			# using one-hot
			# self.correct_prediction = tf.equal(tf.argmax(self.prediction, axis=3), tf.argmax(self.cast_label, axis=3))

			# not using one-hot
			self.correct_prediction = \
				tf.equal(tf.argmax(input=self.prediction, axis=3, output_type=tf.int32), self.input_label)
			self.correct_prediction = tf.cast(self.correct_prediction, tf.float32)
			self.accuracy = tf.reduce_mean(self.correct_prediction)

		# Gradient Descent
		with tf.name_scope('Gradient_Descent'):
			self.train_step = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(self.loss_all)

	def train(self):
		# import cv2
		# import numpy as np
		# ckpt_path = os.path.join(FLAGS.model_dir, "model.ckpt")
		# all_parameters_saver = tf.train.Saver()
		# # import numpy as np
		# # mydata = DataProcess(INPUT_IMG_HEIGHT, INPUT_IMG_WIDE)
		# # imgs_train, imgs_mask_train = mydata.load_my_train_data()
		# my_set_image = cv2.imread('../data_set/train.tif', flags=0)
		# my_set_label = cv2.imread('../data_set/label.tif', flags=0)
		# my_set_image.astype('float32')
		# my_set_label[my_set_label <= 128] = 0
		# my_set_label[my_set_label > 128] = 1
		# my_set_image = np.reshape(a=my_set_image, newshape=(1, INPUT_IMG_WIDE, INPUT_IMG_HEIGHT, INPUT_IMG_CHANNEL))
		# my_set_label = np.reshape(a=my_set_label, newshape=(1, OUTPUT_IMG_WIDE, OUTPUT_IMG_HEIGHT))
		# # cv2.imshow('image', my_set_image)
		# # cv2.imshow('label', my_set_label * 100)
		# # cv2.waitKey(0)
		# with tf.Session() as sess:  # 开始一个会话
		# 	sess.run(tf.global_variables_initializer())
		# 	sess.run(tf.local_variables_initializer())
		# 	for epoch in range(10):
		# 		lo, acc = sess.run(
		# 			[self.loss_mean, self.accuracy],
		# 			feed_dict={
		# 				self.input_image: my_set_image, self.input_label: my_set_label,
		# 				self.keep_prob: 1.0, self.lamb: 0.004}
		# 		)
		# 		# print('num %d, loss: %.6f and accuracy: %.6f' % (epoch, lo, acc))
		# 		print('num %d, loss: %.6f and accuracy: %.6f' % (epoch, lo, acc))
		# 		sess.run(
		# 			[self.train_step],
		# 			feed_dict={
		# 				self.input_image: my_set_image, self.input_label: my_set_label,
		# 				self.keep_prob: 0.6, self.lamb: 0.004}
		# 		)
		# 	all_parameters_saver.save(sess=sess, save_path=ckpt_path)
		# print("Done training")
		train_file_path = os.path.join(TFRECORD_PATH, TRAIN_SET_NAME)
		train_image_filename_queue = tf.train.string_input_producer(
			string_tensor=[train_file_path], num_epochs=EPOCH_NUM, shuffle=True)
		train_images, train_labels = read_image_batch(train_image_filename_queue, TRAIN_BATCH_SIZE)
		tf.summary.scalar("loss", self.loss_mean)
		tf.summary.scalar('accuracy', self.accuracy)
		merged_summary = tf.summary.merge_all()
		all_parameters_saver = tf.train.Saver()
		with tf.Session() as sess:  # 开始一个会话
			sess.run(tf.global_variables_initializer())
			sess.run(tf.local_variables_initializer())
			summary_writer = tf.summary.FileWriter(LOG_DIR, sess.graph)
			coord = tf.train.Coordinator()
			threads = tf.train.start_queue_runners(coord=coord)
			epoch =0
			try:
				while not coord.should_stop():
						# Run training steps or whatever
					epoch+=1
					example, label = sess.run([train_images, train_labels])  # 在会话中取出image和label
					#print(label)

					sess.run([self.train_step],
						feed_dict={
							self.input_image: example, self.input_label: label, self.keep_prob: 0.6,
							self.lamb: 0.004}
						)
					if epoch%10==0:
						lo, acc, summary_str = sess.run(
						[self.loss_mean, self.accuracy, merged_summary],
						feed_dict={
							self.input_image: example, self.input_label: label, self.keep_prob: 1.0,
							self.lamb: 0.004})
						summary_writer.add_summary(summary_str, epoch)
						all_parameters_saver.save(sess=sess, save_path=MODEL_DIR, global_step=epoch)

						print('num %d, loss: %.6f and accuracy: %.6f' % (epoch, lo, acc))
			except tf.errors.OutOfRangeError:
				print('Done training -- epoch limit reached')
			finally:
				# When done, ask the threads to stop.
				coord.request_stop()
			# coord.request_stop()
		coord.join(threads)
		print("Done training")

	def validate(self):
		# import cv2
		# import numpy as np
		# ckpt_path = os.path.join(FLAGS.model_dir, "model.ckpt")
		# # mydata = DataProcess(INPUT_IMG_HEIGHT, INPUT_IMG_WIDE)
		# # imgs_train, imgs_mask_train = mydata.load_my_train_data()
		# all_parameters_saver = tf.train.Saver()
		# my_set_image = cv2.imread('../data_set/train.tif', flags=0)
		# my_set_label = cv2.imread('../data_set/label.tif', flags=0)
		# my_set_image.astype('float32')
		# my_set_label[my_set_label <= 128] = 0
		# my_set_label[my_set_label > 128] = 1
		# with tf.Session() as sess:
		# 	sess.run(tf.global_variables_initializer())
		# 	sess.run(tf.local_variables_initializer())
		# 	all_parameters_saver.restore(sess=sess, save_path=ckpt_path)
		# 	image, acc = sess.run(
		# 		fetches=[self.prediction, self.accuracy],
		# 		feed_dict={
		# 				self.input_image: my_set_image, self.input_label: my_set_label,
		# 				self.keep_prob: 1.0, self.lamb: 0.004}
		# 	)
		# image = np.argmax(a=image[0], axis=2).astype('uint8') * 255
		# # cv2.imshow('predict', image)
		# # cv2.imshow('o', np.asarray(a=image[0], dtype=np.uint8) * 100)
		# # cv2.waitKey(0)
		# cv2.imwrite(filename=os.path.join(FLAGS.model_dir, 'predict.jpg'), img=image)
		# print(acc)
		# print("Done test, predict image has been saved to %s" % (os.path.join(FLAGS.model_dir, 'predict.jpg')))
		validation_file_path = os.path.join('../data_set/my_set', VALIDATION_SET_NAME)
		validation_image_filename_queue = tf.train.string_input_producer(
			string_tensor=[validation_file_path], num_epochs=1, shuffle=True)
		ckpt_path = CHECK_POINT_PATH
		validation_images, validation_labels = read_image_batch(validation_image_filename_queue, VALIDATION_BATCH_SIZE)
		# tf.summary.scalar("loss", self.loss_mean)
		# tf.summary.scalar('accuracy', self.accuracy)
		# merged_summary = tf.summary.merge_all()
		all_parameters_saver = tf.train.Saver()
		with tf.Session() as sess:  # 开始一个会话
			sess.run(tf.global_variables_initializer())
			sess.run(tf.local_variables_initializer())
			# summary_writer = tf.summary.FileWriter(FLAGS.tb_dir, sess.graph)
			# tf.summary.FileWriter(FLAGS.model_dir, sess.graph)
			all_parameters_saver.restore(sess=sess, save_path=MODEL_DIR)
			coord = tf.train.Coordinator()
			threads = tf.train.start_queue_runners(coord=coord)
			try:
				epoch = 1
				while not coord.should_stop():
					# Run training steps or whatever
					# print('epoch ' + str(epoch))
					example, label = sess.run([validation_images, validation_labels])  # 在会话中取出image和label
					# print(label)
					lo, acc = sess.run(
						[self.loss_mean, self.accuracy],
						feed_dict={
							self.input_image: example, self.input_label: label, self.keep_prob: 1.0,
							self.lamb: 0.004}
					)
					# summary_writer.add_summary(summary_str, epoch)
					# print('num %d, loss: %.6f and accuracy: %.6f' % (epoch, lo, acc))
					if epoch % 1 == 0:
						print('num %d, loss: %.6f and accuracy: %.6f' % (epoch, lo, acc))
					epoch += 1
			except tf.errors.OutOfRangeError:
				print('Done validating -- epoch limit reached')
			finally:
				# When done, ask the threads to stop.
				coord.request_stop()
			# coord.request_stop()
			coord.join(threads)
		print('Done validating')

	def test(self):
		import cv2
		test_file_path = os.path.join('../data_set/my_set', TEST_SET_NAME)
		test_image_filename_queue = tf.train.string_input_producer(
			string_tensor=[test_file_path], num_epochs=1, shuffle=True)
		ckpt_path = 'F:/unet/data_set/model/model.ckpt'
		test_images, test_labels = read_image_batch(test_image_filename_queue, TEST_BATCH_SIZE)
		# tf.summary.scalar("loss", self.loss_mean)
		# tf.summary.scalar('accuracy', self.accuracy)
		# merged_summary = tf.summary.merge_all()
		all_parameters_saver = tf.train.Saver()
		with tf.Session() as sess:  # 开始一个会话
			sess.run(tf.global_variables_initializer())
			sess.run(tf.local_variables_initializer())
			# summary_writer = tf.summary.FileWriter(FLAGS.tb_dir, sess.graph)
			# tf.summary.FileWriter(FLAGS.model_dir, sess.graph)
			all_parameters_saver.restore(sess=sess, save_path=ckpt_path)
			coord = tf.train.Coordinator()
			threads = tf.train.start_queue_runners(coord=coord)
			sum_acc = 0.0
			try:
				epoch = 0
				while not coord.should_stop():
					# Run training steps or whatever
					# print('epoch ' + str(epoch))
					example, label = sess.run([test_images, test_labels])  # 在会话中取出image和label
					# print(label)
					image, acc = sess.run(
						[tf.argmax(input=self.prediction, axis=3), self.accuracy],
						feed_dict={
							self.input_image: example, self.input_label: label,
							self.keep_prob: 1.0, self.lamb: 0.004
						}
					)
					sum_acc += acc
					epoch += 1
					cv2.imwrite(os.path.join(PREDICT_SAVED_DIRECTORY, '%d.jpg' % epoch), image[0] * 255)
					if epoch % 1 == 0:
						print('num %d accuracy: %.6f' % (epoch, acc))
			except tf.errors.OutOfRangeError:
				print('Done testing -- epoch limit reached \n Average accuracy: %.2f%%' % (sum_acc / TEST_SET_SIZE * 100))
			finally:
				# When done, ask the threads to stop.
				coord.request_stop()
			# coord.request_stop()
			coord.join(threads)
		print('Done testing')

	def predict(self):
		import cv2
		import glob
		import numpy as np
		# TODO 不应该这样写，应该直接读图片预测，而不是从tfrecord读取，因为顺序变了，无法对应
		predict_file_path = glob.glob(os.path.join(ORIGIN_PREDICT_DIRECTORY, '*.tif'))
		print(len(predict_file_path))
		if not os.path.lexists(PREDICT_SAVED_DIRECTORY):
			os.mkdir(PREDICT_SAVED_DIRECTORY)
		ckpt_path = CHECK_POINT_PATH
		all_parameters_saver = tf.train.Saver()
		with tf.Session() as sess:  # 开始一个会话
			sess.run(tf.global_variables_initializer())
			sess.run(tf.local_variables_initializer())
			# summary_writer = tf.summary.FileWriter(FLAGS.tb_dir, sess.graph)
			# tf.summary.FileWriter(FLAGS.model_dir, sess.graph)
			all_parameters_saver.restore(sess=sess, save_path=ckpt_path)
			for index, image_path in enumerate(predict_file_path):
				# image = cv2.imread(image_path, flags=0)
				image = np.reshape(a=cv2.imread(image_path, flags=0), newshape=(1, INPUT_IMG_WIDE, INPUT_IMG_HEIGHT, INPUT_IMG_CHANNEL))
				predict_image = sess.run(
					tf.argmax(input=self.prediction, axis=3),
					feed_dict={
						self.input_image: image,
						self.keep_prob: 1.0, self.lamb: 0.004
					}
				)
				cv2.imwrite(os.path.join(PREDICT_SAVED_DIRECTORY, '%d.jpg' % index), predict_image[0] * 255)
		print('Done prediction')


def main():
	net = Unet()
	CHECK_POINT_PATH = os.path.join(MODEL_DIR, "model.ckpt")
	#net.set_up_unet(TRAIN_BATCH_SIZE)
	#net.train()
	#net.set_up_unet(VALIDATION_BATCH_SIZE)
	#net.validate()
	net.set_up_unet(TEST_BATCH_SIZE)
	net.test()
	# net.set_up_unet(PREDICT_BATCH_SIZE)
	# net.predict()

if __name__ == '__main__':
	#read_check_tfrecords()#用来检查tfrecord中的数据，显示image和label图像
	main()
