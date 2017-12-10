import os  
import numpy as np  
import tensorflow as tf  
import input_data     
import model  

  
N_CLASSES = 2  # 2个输出神经元，［1，0］ 或者 ［0，1］猫和狗的概率
IMG_W = 208  # 重新定义图片的大小，图片如果过大则训练比较慢  
IMG_H = 208  
BATCH_SIZE = 32  #每批数据的大小
CAPACITY = 256  
MAX_STEP = 15000 # 训练的步数，应当 >= 10000
learning_rate = 0.0001 # 学习率，建议刚开始的 learning_rate <= 0.0001
  

def run_training():  
      
    # 数据集
    train_dir = '/Users/yangyibo/GitWork/pythonLean/AI/猫狗识别/img/'   #My dir--20170727-csq  
    #logs_train_dir 存放训练模型的过程的数据，在tensorboard 中查看 
    logs_train_dir = '/Users/yangyibo/GitWork/pythonLean/AI/猫狗识别/saveNet/'  

    # 获取图片和标签集
    train, train_label = input_data.get_files(train_dir)  
    # 生成批次
    train_batch, train_label_batch = input_data.get_batch(train,  
                                                          train_label,  
                                                          IMG_W,  
                                                          IMG_H,  
                                                          BATCH_SIZE,   
                                                          CAPACITY)
    # 进入模型
    train_logits = model.inference(train_batch, BATCH_SIZE, N_CLASSES) 
    # 获取 loss 
    train_loss = model.losses(train_logits, train_label_batch)
    # 训练 
    train_op = model.trainning(train_loss, learning_rate)
    # 获取准确率 
    train__acc = model.evaluation(train_logits, train_label_batch)  
    # 合并 summary
    summary_op = tf.summary.merge_all()  
    sess = tf.Session()
    # 保存summary
    train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)  
    saver = tf.train.Saver()  
      
    sess.run(tf.global_variables_initializer())  
    coord = tf.train.Coordinator()  
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)  
      
    try:  
        for step in np.arange(MAX_STEP):  
            if coord.should_stop():  
                    break  
            _, tra_loss, tra_acc = sess.run([train_op, train_loss, train__acc])  
                 
            if step % 50 == 0:  
                print('Step %d, train loss = %.2f, train accuracy = %.2f%%' %(step, tra_loss, tra_acc*100.0))  
                summary_str = sess.run(summary_op)  
                train_writer.add_summary(summary_str, step)  
              
            if step % 2000 == 0 or (step + 1) == MAX_STEP:  
                # 每隔2000步保存一下模型，模型保存在 checkpoint_path 中
                checkpoint_path = os.path.join(logs_train_dir, 'model.ckpt')  
                saver.save(sess, checkpoint_path, global_step=step)  
                  
    except tf.errors.OutOfRangeError:  
        print('Done training -- epoch limit reached')  
    finally:  
        coord.request_stop()
    coord.join(threads)  
    sess.close()  

# train
run_training()
 