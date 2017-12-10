# -*- coding: utf-8 -*-
# save 训练好的神经网络，以便下次使用
# tensorflow 目前还不能保存 整个 神经网络框架只能保存我门的 Variable
import tensorflow as tf
import numpy as np


# save Variable

# 两行 三列 的 Weights，，最好定义一下 dtype 一般都是 tf.float32 name 又没有都可以
# 记得定义 形状 和 dtype 在导入的时候
# W = tf.Variable([[1,2,3],[3,4,5]], dtype=tf.float32, name='weights')

# b = tf.Variable([[1,2,3]], dtype=tf.float32, name='biases')


# init = tf.global_variables_initializer()

# # 定义saver
# saver = tf.train.Saver()

# with tf.Session() as sess:
# 	sess.run(init)
# 	# 保存的是整个session 中的东西, 保存到my_net/save_net.ckpt ，官方默认后缀 ckpt
# 	save_path = saver.save(sess,"my_net/save_net.ckpt")
# 	print("save_path: ",save_path)



# restore Variable
# 重新定义和保存的 网络 相同 形状和类型的 网络才能正确导入


# 形状是两行 三列的 的矩阵 dtype=tf.float32
W = tf.Variable(np.arange(6).reshape((2, 3)), dtype=tf.float32, name="weights")
# 形状是一行 三列的 的矩阵 dtype=tf.float32
b = tf.Variable(np.arange(3).reshape((1,3)), dtype=tf.float32, name='biases')

# 在restore 的时候不用 init 了
saver = tf.train.Saver()

with tf.Session() as sess:
	saver.restore(sess,"my_net/save_net.ckpt")
	print("weights: ",sess.run(W))
	print("biases: ",sess.run(b))

