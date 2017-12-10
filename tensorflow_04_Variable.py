# -*- coding: utf-8 -*-
# 目标 Tensorflow 中使用 Variable 定义了某字符串是变量，它才是变量，这一点是与 Python 所不同的。
# 定义语法： state = tf.Variable()
import tensorflow as tf

state = tf.Variable(0,name='counter')
print(state.name)
# 定义一个常量
one = tf.constant(1)

# 定义加法步骤 (注: 此步并没有直接计算)
new_value = tf.add(state, one)
 
# 更新 new_value 加载到了  state 也就是  state ＝ new_value 
update = tf.assign( state, new_value)

# 如果定义 Variable, 就一定要 initialize 激活变量 （此时没有激活只有 sess.run 才会激活）
init = tf.global_variables_initializer()  

with tf.Session() as sess:
	sess.run(init)
	# 做3次循环
	for _ in range(3):
		sess.run(update);
		# print(state) 是无效的。一定要把 sess 的指针指向 state 再进行 print 才能得到想要的结果！
		print(sess.run(state)) 