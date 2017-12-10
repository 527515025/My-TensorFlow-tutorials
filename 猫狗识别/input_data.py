import tensorflow as tf
import os 
import numpy as np

def get_files(file_dir):
	cats = []
	label_cats = []
	dogs = []
	label_dogs = []
	for file in os.listdir(file_dir):
		name = file.split(sep='.')
		if 'cat' in name[0]:
			cats.append(file_dir + file)
			label_cats.append(0)
		else:
			if 'dog' in name[0]:
				dogs.append(file_dir + file)
				label_dogs.append(1)
		image_list = np.hstack((cats,dogs))
		label_list = np.hstack((label_cats,label_dogs))
			# print('There are %d cats\nThere are %d dogs' %(len(cats), len(dogs)))
			# 多个种类分别的时候需要把多个种类放在一起，打乱顺序,这里不需要
     
	# 把标签和图片都放倒一个 temp 中 然后打乱顺序，然后取出来
	temp = np.array([image_list,label_list])
	temp = temp.transpose()
	# 打乱顺序
	np.random.shuffle(temp)

	# 取出第一个元素作为 image 第二个元素作为 label
	image_list = list(temp[:,0])
	label_list = list(temp[:,1])
	label_list = [int(i) for i in label_list]  
	return image_list,label_list

# 测试 get_files
# imgs , label = get_files('/Users/yangyibo/GitWork/pythonLean/AI/猫狗识别/testImg/')
# for i in imgs:
# 	print("img:",i)

# for i in label:
# 	print('label:',i)
# 测试 get_files end


# image_W ,image_H 指定图片大小，batch_size 每批读取的个数 ，capacity队列中 最多容纳元素的个数
def get_batch(image,label,image_W,image_H,batch_size,capacity):
	# 转换数据为 ts 能识别的格式
	image = tf.cast(image,tf.string)
	label = tf.cast(label, tf.int32)

	# 将image 和 label 放倒队列里 
	input_queue = tf.train.slice_input_producer([image,label])
	label = input_queue[1]
	# 读取图片的全部信息
	image_contents = tf.read_file(input_queue[0])
	# 把图片解码，channels ＝3 为彩色图片, r，g ，b  黑白图片为 1 ，也可以理解为图片的厚度
	image = tf.image.decode_jpeg(image_contents,channels =3)
	# 将图片以图片中心进行裁剪或者扩充为 指定的image_W，image_H
	image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)
	# 对数据进行标准化,标准化，就是减去它的均值，除以他的方差
	image = tf.image.per_image_standardization(image)

	# 生成批次  num_threads 有多少个线程根据电脑配置设置  capacity 队列中 最多容纳图片的个数  tf.train.shuffle_batch 打乱顺序，
	image_batch, label_batch = tf.train.batch([image, label],batch_size = batch_size, num_threads = 64, capacity = capacity)
	
    # 重新定义下 label_batch 的形状
	label_batch = tf.reshape(label_batch , [batch_size])
	# 转化图片
	image_batch = tf.cast(image_batch,tf.float32)
	return  image_batch, label_batch
  

# test get_batch
# import matplotlib.pyplot as plt
# BATCH_SIZE = 2
# CAPACITY = 256  
# IMG_W = 208
# IMG_H = 208

# train_dir = '/Users/yangyibo/GitWork/pythonLean/AI/猫狗识别/testImg/'

# image_list, label_list = get_files(train_dir)
# image_batch, label_batch = get_batch(image_list, label_list, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)

# with tf.Session() as sess:
#    i = 0
#    #  Coordinator  和 start_queue_runners 监控 queue 的状态，不停的入队出队
#    coord = tf.train.Coordinator()
#    threads = tf.train.start_queue_runners(coord=coord)
#    # coord.should_stop() 返回 true 时也就是 数据读完了应该调用 coord.request_stop()
#    try: 
#        while not coord.should_stop() and i<1:
#            # 测试一个步
#            img, label = sess.run([image_batch, label_batch])
           
#            for j in np.arange(BATCH_SIZE):
#                print('label: %d' %label[j])
#                # 因为是个4D 的数据所以第一个为 索引 其他的为冒号就行了
#                plt.imshow(img[j,:,:,:])
#                plt.show()
#            i+=1
#    # 队列中没有数据
#    except tf.errors.OutOfRangeError:
#        print('done!')
#    finally:
#        coord.request_stop()
#    coord.join(threads)
   # sess.close()


