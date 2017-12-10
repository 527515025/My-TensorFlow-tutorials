# -*- coding: utf-8 -*-
# 学会用 Tensorflow 自带的 tensorboard 去可视化我们所建造出来的神经网络是一个很好的学习理解方式. 用最直观的流程图告诉你你的神经网络是长怎样,有助于你发现编程中间的问题和疑问.
# 定义一个神经层，用于图表展示  使用 tensorboard 去展示神经网络学习的过程，从而用来优化神经网络
import tensorflow as tf

import numpy as np

import matplotlib.pyplot as plt


# 添加神经层的函数def add_layer(),它有四个参数：输入值、输入的大小、输出的大小和激励函数，我们设定默认的激励函数是None。也就是线性函数
def add_layer(inputs, in_size, out_size, n_layer,  activation_function=None):
    layer_name='layer%s'%n_layer  ## define a new var
    with tf.name_scope('layer'):
        with tf.name_scope('Weights'):
	    # 定义权重,尽量是一个随机变量
	    # 因为在生成初始参数时，随机变量(normal distribution) 会比全部为0要好很多，所以我们这里的weights 是一个 in_size行, out_size列的随机变量矩阵。   
            Weights = tf.Variable(tf.random_normal([in_size, out_size]),name='W')
            # tf.histogram_summary(layer_name+'/weights',Weights)   # tensorflow 0.12 以下版的
            # 放在 histogram 里面则在 tensorboard 的 histogram 中展示
            tf.summary.histogram(layer_name + '/weights', Weights) # tensorflow >= 0.12
        with tf.name_scope('biases'):
            # 在机器学习中，biases的推荐值不为0，所以我们这里是在0向量的基础上又加了0.1。
            biases = tf.Variable(
            tf.zeros([1, out_size]) + 0.1, 
            name='b')
            tf.summary.histogram(layer_name + '/biases', biases)
        with tf.name_scope('Wx_plus_b'):
            # 定义Wx_plus_b, 即神经网络未激活的值(预测的值)。其中，tf.matmul()是矩阵的乘法。
            Wx_plus_b = tf.add(tf.matmul(inputs, Weights),biases)
    	# activation_function ——激励函数（激励函数是非线性方程）为None时(线性关系)，输出就是当前的预测值——Wx_plus_b，
    	# 不为None时，就把Wx_plus_b传到activation_function()函数中得到输出。
        if activation_function is None:
            outputs = Wx_plus_b
        else:
        	# 返回输出
            outputs = activation_function(Wx_plus_b)
            tf.summary.histogram(layer_name + '/outputs', outputs)
        return outputs

	# 添加一个神经层的函数——def add_layer()就定义好了。


# 数据准备
# -1到1这个区间，有300个单位，[:,np.newaxis] 加纬度，x_data 一个特性有300个例子
x_data = np.linspace(-1,1,300,dtype=np.float32)[:,np.newaxis] 
# 噪点，使点分布在 线性方程的线的两边，使数据看起来更加真实 ,他的幂是0 方差是0.05，格式 是x_data 一样的格式
noise = np.random.normal(0, 0.05, x_data.shape).astype(np.float32)
# np.square(x_data) x_data 的二次方
y_data = np.square(x_data) - 0.5 + noise




#定义一个大的框架，包含x_input和 y_input
with tf.name_scope('inputs'):
# 输入层 输入多少个data  输入层就有多少个神经元
# 输入属性和输出属性都是 1 ，None 指给出多少个例子都可以
    xs = tf.placeholder(tf.float32, [None, 1],name='x_input')
    ys = tf.placeholder(tf.float32, [None, 1],name='y_input')



# add hidden layer 隐藏层 假设十个神经元
# 输入 x_data ，x_data的size＝1  ，out_size=10
l1 = add_layer(xs, 1, 10,n_layer=1,activation_function=tf.nn.relu)

# add output layer 输出层 有一个输出 所以一个神经元
# 从隐藏层拿到数据 放入add_layer 执行。 数据是 l1 size ＝10 （因为 隐藏层的out_size=10） ，out_size＝1 激励函数为空
prediction = add_layer(l1, 10, 1,n_layer=2, activation_function=None)

with tf.name_scope('loss'):
    #predition和真实值的偏差  将10个例子的每个例子的结果都减去predition取平方，然后再求和,然后取平均
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
                     reduction_indices=[1]))
    #当loss 越来越小的时候，说明这个神经网络学到东西了，如果loss 稳定了，说明这个神经网络已经不学习了
    # tf.scalar_summary('loss',loss) # tensorflow < 0.12
    # 放在 scalar 里面则在 tensorboard 的scalars 中展示
    tf.summary.scalar('loss', loss) # tensorflow >= 0.12

with tf.name_scope('train_step'):
    # 优化器 如何让机器学习提升它的准确率。 tf.train.GradientDescentOptimizer()中的值（学习的效率）通常都小于1，这里取的是0.1，代表以0.1的效率来最小化（minimize 减小）误差loss。
    # 每一个练习的步骤都通过这个优化器 以学习进度0.1的效率 对误差进行更正和提升，下一次就有更好的结果。
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)


# 定义session
sess = tf.Session()

#给所有训练图合并
# merged = tf.merge_all_summaries() # tensorflow < 0.12
merged = tf.summary.merge_all() # tensorflow >= 0.12

#把整个框架加载到一个文件中去
writer = tf.summary.FileWriter("/Users/yangyibo/test/logs/",sess.graph)
# 执行init
sess.run(tf.initialize_all_variables())

for i in range(1000):
    sess.run(train_step,feed_dict={xs: x_data, ys: y_data})
    if i%50 == 0:
        result = sess.run(merged,feed_dict={xs: x_data, ys: y_data})
        writer.add_summary(result,i)


# 在 /Users/yangyibo/test/ 目录下 执行   这个目录是我门框架加载到的那个文件的路径
# tensorboard --logdir logs

# 然后 chrome 浏览器打开 http://localhost:6006  
# 然后选择 scalar 或 histogram tab 就可以了。 图像中颜色越深的区域值的个数越多，颜色越浅的地方值个数越少

 



