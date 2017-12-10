# -*- coding: utf-8 -*-
# 目标 学习session  ，输出两个矩阵想乘的结果
import tensorflow as tf

# 一个一行两列的矩阵
matrix1 = tf.constant([[3,3]])
# 一个两行一列的矩阵
matrix2 = tf.constant([[2],
                       [2]])

# 矩阵想乘 3*2 ＋ 3*2 ＝12 类似 np 的 np.dot(m1,m2)
product = tf.matmul(matrix1,matrix2)

# method 1
sess = tf.Session()
 # 执行 product 计算 结构
result = sess.run(product)
print(result)
sess.close()

# method 2 
# 打开  tf.Session() 以  sess 命名 不用管 sess.close() 在运行最后会自动关闭
with tf.Session() as sess:

	result2 = sess.run(product)
	print(result2)
print('abel')
