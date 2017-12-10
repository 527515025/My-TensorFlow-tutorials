# -*- coding: utf-8 -*-
# 定义一个神经层
import tensorflow as tf

# 添加神经层的函数def add_layer(),它有四个参数：输入值、输入的大小、输出的大小和激励函数，我们设定默认的激励函数是None。也就是线性函数
def add_layer(inputs, in_size, out_size, activation_function=None):
	# 定义权重,尽量是一个随机变量
	# 因为在生成初始参数时，随机变量(normal distribution) 会比全部为0要好很多，所以我们这里的weights 是一个 in_size行, out_size列的随机变量矩阵。   
	Weights = tf.Variable(tf.random_normal([in_size,out_size]))
	# 在机器学习中，biases的推荐值不为0，所以我们这里是在0向量的基础上又加了0.1。
	biases = tf.Variable(tf.zores([1,out_size]) + 0.1)
	# 定义Wx_plus_b, 即神经网络未激活的值(预测的值)。其中，tf.matmul()是矩阵的乘法。
	Wx_plus_b = tf.matmul(inputs,Weights)+biases
	# activation_function ——激励函数（激励函数是非线性方程）为None时(线性关系)，输出就是当前的预测值——Wx_plus_b，
	# 不为None时，就把Wx_plus_b传到activation_function()函数中得到输出。
	if activation_function = None:
		outputs = Wx_plus_b
	else:
		outputs=activation_function(Wx_plus_b)
	# 返回输出
	return outputs

	# 添加一个神经层的函数——def add_layer()就定义好了。



