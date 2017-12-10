# -*- coding: utf-8 -*-
# 目标 Tensorflow 中的 placeholder , placeholder 是 Tensorflow 中的占位符，暂时储存变量.
# Tensorflow 如果想要从外部传入data, 那就需要用到 tf.placeholder(), 然后以这种形式传输数据 sess.run(***, feed_dict={input: **}).
import tensorflow as tf


#在 Tensorflow 中需要定义 placeholder 的 type ，一般为 float32 形式
# tf.placeholder(tf.float32,[2,2]) 规定数据结构
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)

# mul = multiply 是将input1和input2 做乘法运算，并输出为 output 
ouput = tf.multiply(input1, input2)

with tf.Session() as sess:
	# 以feed_dict 传入参数，python 字典的格式   
	# 需要传入的值放在了feed_dict={} 并一一对应每一个 input. placeholder 与 feed_dict={} 是绑定在一起出现的。
	print(sess.run(ouput,feed_dict={input1: [7.], input2: [2.]}))