# -*- coding: utf-8 -*-
# 手写数字识别
# 准确度 0.8802
import input_data  
import tensorflow as tf
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)  

# 添加神经层的函数def add_layer(),它有四个参数：输入值、输入的大小、输出的大小和激励函数，我们设定默认的激励函数是None。也就是线性函数
def add_layer(inputs, in_size, out_size, activation_function=None):
	# 定义权重,尽量是一个随机变量
	# 因为在生成初始参数时，随机变量(normal distribution) 会比全部为0要好很多，所以我们这里的weights 是一个 in_size行, out_size列的随机变量矩阵。   
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
	# 在机器学习中，biases的推荐值不为0，所以我们这里是在0向量的基础上又加了0.1。
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
	# 定义Wx_plus_b, 即神经网络未激活的值(预测的值)。其中，tf.matmul()是矩阵的乘法。
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
	# activation_function ——激励函数（激励函数是非线性方程）为None时(线性关系)，输出就是当前的预测值——Wx_plus_b，
	# 不为None时，就把Wx_plus_b传到activation_function()函数中得到输出。
    if activation_function is None:
        outputs = Wx_plus_b
    else:
    	# 返回输出
        outputs = activation_function(Wx_plus_b)
    return outputs

	# 添加一个神经层的函数——def add_layer()就定义好了。


def compute_accuracy(v_xs,v_ys):
	# 定义 prediction 为全局变量
	global prediction
	# 将 xs data 在 prediction 中生成预测值，预测值也是一个 1*10 的矩阵 中每个值的概率，并不是一个0-9 的值，是0-9 每个值的概率 ，比如说3这个位置的概率最高，那么预测3就是这个图片的值
	y_pre = sess.run(prediction, feed_dict={xs: v_xs})
	# 对比我的预测值y_pre 和真实数据 v_ys 的差别
	correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
	# 计算我这一组数据中有多少个预测是对的，多少个是错的
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	# result 是一个百分比，百分比越高，预测越准确
	result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})
	return result   

xs = tf.placeholder(tf.float32,[None, 784]) #图像输入向量  每个图片有784 （28 ＊28） 个像素点
ys = tf.placeholder(tf.float32, [None,10]) #每个例子有10 个输出

prediction = add_layer(xs, 784, 10, activation_function=tf.nn.softmax)

#loss函数（即最优化目标函数）选用交叉熵函数。交叉熵用来衡量预测值和真实值的相似程度，如果完全相同，它们的交叉熵等于零 ,所以loss 越小 学的好
#分类一般都是 softmax＋ cross_entropy
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
reduction_indices=[1]))
# cross_entropy = -tf.reduce_sum(ys*tf.log(prediction))  

#train方法（最优化算法）采用梯度下降法。  优化器 如何让机器学习提升它的准确率。 tf.train.GradientDescentOptimizer()中的值（学习的效率）通常都小于1
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
sess = tf.Session()

# 初始化变量
init= tf.global_variables_initializer()
sess.run(init)

for i in range(1000):
	#开始train，每次只取100张图片，免得数据太多训练太慢
	batch_xs, batch_ys = mnist.train.next_batch(50)
	sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys})
	if i % 50 == 0:
		print(compute_accuracy(
            mnist.test.images, mnist.test.labels))