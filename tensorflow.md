##TensorFlow 
TensorFlow 是一个编程系统, 使用图来表示计算任务. 图中的节点被称之为 op (operation 的缩写). 一个 op 获得 0 个或多个 Tensor, 执行计算, 产生 0 个或多个 Tensor. 每个 Tensor 是一个类型化的多维数组. 例如, 你可以将一小组图像集表示为一个四维浮点数数组, 这四个维度分别是 [batch, height, width, channels].

一个 TensorFlow 图描述了计算的过程. 为了进行计算, 图必须在 会话 里被启动. 会话 将图的 op 分发到诸如 CPU 或 GPU 之类的 设备 上, 同时提供执行 op 的方法. 这些方法执行后, 将产生的 tensor 返回. 在 Python 语言中, 返回的 tensor 是 numpy ndarray 对象; 在 C 和 C++ 语言中, 返回的 tensor 是 tensorflow::Tensor 实例.



TensorFlow的一个基本的功能，就是支持在线数据不断优化模型。TensorFlow可以通过tf.train.Saver()来保存模型和恢复模型参数，使用Python加载模型文件后，可不断接受在线请求的数据，更新模型参数后通过Saver保存成checkpoint，用于下一次优化或者线上服务。



#Session
神经网络 就是一个图片

 Session 是 Tensorflow 为了控制,和输出文件的执行的语句. 运行 session.run() 可以获得你要得知的运算结果, 或者是你所要运算的部分.
 
 session用来run() 我门设计好的结构，神经网络图片上的一个小结构，一个小功能，一个小图片，输出运行的结构的结果，如果指向一个参数的话就会输出当前参数值，如果指向一个运算的话就会输出运算的值
 

#Tensor

TensorFlow 程序使用 tensor 数据结构来代表所有的数据, 计算图中, 操作间传递的数据都是 tensor. 你可以把 TensorFlow tensor 看作是一个 n 维的数组或列表. 一个 tensor 包含一个静态类型 rank, 和 一个 shape. 
#变量
Variables for more details. 变量维护图执行过程中的状态信息. 下面的例子演示了如何使用变量实现一个简单的计数器. 


#placeholder
也就是在sess.run()的时候再输入计算需要的值

 placeholder 是 Tensorflow 中的占位符，暂时储存变量.Tensorflow 如果想要从外部传入data, 那就需要用到 tf.placeholder(), 然后以这种形式传输数据 sess.run(***, feed_dict={input: **}).
 
 需要传入的值放在了feed_dict={} 并一一对应每一个 input. placeholder 与 feed_dict={} 是绑定在一起出现的
 
#激励函数 activation function
 激励函数运行时激活神经网络中某一部分神经元，将激活信息向后传入下一层的神经系统。激励函数的实质是非线性方程。 Tensorflow 的神经网络 里面处理较为复杂的问题时都会需要运用激励函数 activation function 。
 
##分布式TensorFlow
分布式TensorFlow中ps、worker、in-graph、between-graph、synchronous training和asynchronous training的概念

ps是整个训练集群的参数服务器，保存模型的Variable

worker是计算模型梯度的节点，得到的梯度向量会交付给ps更新模型

in-graph与between-graph对应，但两者都可以实现同步训练和异步训练，in-graph指整个集群由一个client来构建graph，并且由这个client来提交graph到集群中，其他worker只负责处理梯度计算的任务，而between-graph指的是一个集群中多个worker可以创建多个graph，但由于worker运行的代码相同因此构建的graph也相同，并且参数都保存到相同的ps中保证训练同一个模型，这样多个worker都可以构建graph和读取训练数据，适合大数据场景

同步训练和异步训练差异在于，同步训练每次更新梯度需要阻塞等待所有worker的结果，而异步训练不会有阻塞，训练的效率更高，在大数据和分布式的场景下一般使用异步训练。


在一幅 TensorFlow 图中，每个节点（node）有一个或者多个输入和零个或者多个输出，表示一种操作（operation）的实例化。

##TensorFlow Serving
TensorFlow Serving 能够简化并加速从模型到生产的过程。它能实现在服务器架构和 API 保持不变的情况下，安全地部署新模型并运行试验。除了原生集成 TensorFlow，还可以扩展服务其他类型的模型。

#语法
```
创建一个一行两列的矩阵
matrix1 = tf.constant([[3., 3.]])
创建一个两行一列的矩阵
matrix2 = tf.constant([[2.],[2.]])
矩阵相乘
tf.matmul(matrix1, matrix2)
启动默认图.
sess = tf.Session()
result = sess.run(product)
任务完成, 关闭会话.
sess.close()
创建一个变量列表,(变量维护图的中间状态)
tf.Variable([1.0, 2.0])
创建一个常量列表
tf.constant([3.0, 3.0])
使用初始化器 initializer op 的 run() 方法初始化 'x' 
x.initializer.run()
增加一个减法 sub op, 从 'x' 减去 'a'. 运行减法 op, 输出结果 
sub = tf.sub(x, a)
计算state 和 one 的值
new_value = tf.add(state, one)
将 new_value 的值 赋给 state （state＝new_value）
update = tf.assign(state, new_value)
```


## 
池化层
卷积层

过拟合 drop out
欠拟合



