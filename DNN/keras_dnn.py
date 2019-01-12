#由于不能链接到官方的已经处理好的数据，所以这里通过tensorflow导入mnist数据
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
#from _future_ import print_function
import numpy as np
#from keras.dataseets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils

batch_size = 128 #梯度下降一个批（batch）的数据量
nb_classes =10 #类别
nb_epoch =10 #梯度下降epoch循环训练次数，每次循环包含全部的样本
image_size = 28*28 #输入图片的大小，由于是灰度图片，因此只有一个颜色通道

#加载数据
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
x_train,y_train= mnist.train.images, mnist.train.labels
#print(x_train.shape,y_train.shape) #(55000, 784) (55000, 10)
x_test,y_test= mnist.test.images, mnist.test.labels
#print(x_test.shape,y_test.shape) #(10000, 784) (10000, 10)
#如果y_train\y_test不是one_hot编码，需要进行转换

#创建模型，逻辑分类相当于一层全链接的神经网络（Dense是Keras中定义的DNN模型）
model = Sequential([Dense(128,input_shape=(image_size,),activation= 'relu'),Dense(10,input_shape=(128,),activation= 'softmax')])
#配置优化器，损失函数
model.compile(optimizer = 'rmsprop',loss = 'categorical_crossentropy',metrics= ['accuracy'])
model.fit(x_train,y_train,batch_size = batch_size,nb_epoch = nb_epoch,verbose = 1,validation_data = (x_test,y_test))
#score分数包含两部分，一部分是val_loss,一部分是val_acc。取score[1]来进行模型的得分评价
score = model.evaluate(x_test,y_test,verbose = 0)
print('Accuracy:{}'.format(score[1]))