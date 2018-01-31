#coding=utf-8  
import tensorflow as tf 
from PIL import Image  
import matplotlib.pyplot as plt
import input_data 
import numpy as np
import model
import os 
  
#从训练集中选取一张图片 
def get_one_image(train): 
    files = os.listdir(train)
    n = len(files)
    ind = np.random.randint(0,n)
    img_dir = os.path.join(train,files[ind])  
    image = Image.open(img_dir)  
    plt.imshow(image)
    plt.show()
    image = image.resize([208, 208])  
    image = np.array(image)
    return image  
  
  
def evaluate_one_image():  
    train = '/Users/yangyibo/GitWork/pythonLean/AI/猫狗识别/testImg/'  
  
    # 获取图片路径集和标签集
    image_array = get_one_image(train)  
      
    with tf.Graph().as_default():  
        BATCH_SIZE = 1  # 因为只读取一副图片 所以batch 设置为1
        N_CLASSES = 2  # 2个输出神经元，［1，0］ 或者 ［0，1］猫和狗的概率
        # 转化图片格式
        image = tf.cast(image_array, tf.float32)  
        # 图片标准化
        image = tf.image.per_image_standardization(image)
        # 图片原来是三维的 [208, 208, 3] 重新定义图片形状 改为一个4D  四维的 tensor
        image = tf.reshape(image, [1, 208, 208, 3])  
        logit = model.inference(image, BATCH_SIZE, N_CLASSES)  
        # 因为 inference 的返回没有用激活函数，所以在这里对结果用softmax 激活
        logit = tf.nn.softmax(logit)  
        
        # 用最原始的输入数据的方式向模型输入数据 placeholder
        x = tf.placeholder(tf.float32, shape=[208, 208, 3])  
        
        # 我门存放模型的路径
        logs_train_dir = '/Users/yangyibo/GitWork/pythonLean/AI/猫狗识别/saveNet/'   
        # 定义saver 
        saver = tf.train.Saver()  
          
        with tf.Session() as sess:  
              
            print("从指定的路径中加载模型。。。。")
            # 将模型加载到sess 中 
            ckpt = tf.train.get_checkpoint_state(logs_train_dir)  
            if ckpt and ckpt.model_checkpoint_path:  
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]  
                saver.restore(sess, ckpt.model_checkpoint_path)  
                print('模型加载成功, 训练的步数为 %s' % global_step)  
            else:  
                print('模型加载失败，，，文件没有找到')  
            # 将图片输入到模型计算
            prediction = sess.run(logit, feed_dict={x: image_array})
            # 获取输出结果中最大概率的索引
            max_index = np.argmax(prediction)  
            if max_index==0:  
                print('猫的概率 %.6f' %prediction[:, 0])  
            else:  
                print('狗的概率 %.6f' %prediction[:, 1]) 
# 测试
evaluate_one_image()
