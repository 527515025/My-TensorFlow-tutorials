# -*- coding: utf-8 -*-
# 预测一个线性的直线 ，预测 y = 0.1*x+0.3
import tensorflow as tf
# 导入科学计算模块  
import numpy as np

#自己编一些数据，因为在tensorflow 中，他的大部分数据格式是 float 32 的形式 
x_data = np.random.rand(100).astype(np.float32) 
# 这就是我们预测的 y=Weights * x + biases   Weights 接近0.1  激励 接近0.3 然后神经网络也就是学着把 Weights 变成 0.1, biases 变成 0.3
y_data = x_data*0.1 + 0.3

# 开始创建 tensorflow 的结构  
#  Weights可能是个矩阵，此处定义 Weights 变量是一个一维的 参数范围为 -1.0 到 1.0的变量
Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
# biases 是一个一维变量，初始值是 0 
biases  = tf.Variable(tf.zeros([1]))
# 上边两步是生成两个初始值，Weights 和 biases ，然后 Weights 和 biases 经过学习会越来越趋近于 0.1 和 0.3


# 预测的y 
y = Weights*x_data + biases
# 接着就是计算 y预测值 和 y_data真实值 的误差:
loss = tf.reduce_mean(tf.square(y-y_data))

# 建立一个优化器， 减少神将网络的误差  GradientDescentOptimizer 是最基础的优化器， 0.5 为学习效率（0-1），
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

# 初始化变量 ，神经网络就是一个图，上边就是建立结构，这里是初始化变量，激活图
# init = tf.initialize_all_variables() # tf 马上就要废弃这种写法
init = tf.global_variables_initializer()  # 替换成这样就好


sess = tf.Session()
# 从init 开始 跑我们的图片，init 就是神经网络的入口
sess.run(init) 

       # 训练201次
for step in range(201):
	# 开始训练
    sess.run(train)
    if step % 20 == 0:
     # 每隔20 次训练 输出一下当前我的 Weights参数  和 biases参数 , run(Weights) 就是像指针一样指向我图中的 Weights
        print(step, sess.run(Weights), sess.run(biases))

