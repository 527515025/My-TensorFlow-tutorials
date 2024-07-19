import tensorflow as tf
import os
import numpy as np


def get_files(file_dir_cat, file_dir_dog):
    # cats和label_cats用于存储猫的图像路径和对应的标签（0 表示猫），dogs和label_dogs用于存储狗的图像路径和对应的标签（1 表示狗）
    cats = []
    label_cats = []
    dogs = []
    label_dogs = []
    valid_extensions = ['.jpg', '.jpeg', '.png']

    for file in os.listdir(file_dir_cat):
        if any(file.endswith(ext) for ext in valid_extensions):
            cats.append(os.path.join(file_dir_cat, file))
            label_cats.append(0)

    for file in os.listdir(file_dir_dog):
        if any(file.endswith(ext) for ext in valid_extensions):
            dogs.append(os.path.join(file_dir_dog, file))
            label_dogs.append(1)

    # 将cats和dogs列表合并成一个 image_list，将label_cats和label_dogs列表合并成一个 label_list
    image_list = np.hstack((cats, dogs))
    label_list = np.hstack((label_cats, label_dogs))
    print('There are %d cats\nThere are %d dogs' %(len(cats), len(dogs)))

    # 将image_list和label_list组合成一个二维数组 temp，然后对其进行转置，使得每行包含一个图像路径和一个对应的标签
    temp = np.array([image_list, label_list])
    temp = temp.transpose()
    # 打乱数组顺序
    np.random.shuffle(temp)

    # 从打乱后的数组 temp 中提取图像路径和标签，分别存储在 image_list 和 label_list 中
    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    # 将label_list中的元素转换为整数类型
    label_list = [int(i) for i in label_list]

    return image_list, label_list


# 测试 get_files
file_dir_cat = "C:/Cat-or-Dog/test/cat"
file_dir_dog = "C:/Cat-or-Dog/test/dog"
imgs, label = get_files(file_dir_cat, file_dir_dog)
for i in imgs:
    print("img:", i)

for i in label:
    print('label:', i)


# 读取图像数据，进行预处理（包括解码、调整大小和标准化），并生成图像和标签的批次
# image_W ,image_H 指定图片大小，batch_size 每批读取的个数 ，capacity队列中最多容纳元素的个数
def get_batch(image_paths, labels, image_W, image_H, batch_size, capacity):
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))

    def load_and_preprocess_image(image_path, label):
        image_contents = tf.io.read_file(image_path)
        try:
            image = tf.image.decode_image(image_contents, channels=3, expand_animations=False)

            if image.shape[-1] != 3:
                tf.print(f"Skipping image {image_path}, as it has {image.shape[-1]} channels.")
                return tf.constant(0.0, shape=(image_W, image_H, 3)), tf.constant(-1, dtype=tf.int32)

            image = tf.image.resize_with_crop_or_pad(image, image_W, image_H)
            image = tf.image.per_image_standardization(image)
            return image, label
        except tf.errors.InvalidArgumentError as e:
            tf.print(f"Skipping image {image_path}, due to error: {e}")
            return tf.constant(0.0, shape=(image_W, image_H, 3)), tf.constant(-1, dtype=tf.int32)
        except Exception as e:
            tf.print(f"Skipping image {image_path}, due to an unexpected error: {e}")
            return tf.constant(0.0, shape=(image_W, image_H, 3)), tf.constant(-1, dtype=tf.int32)

    def filter_invalid_images(image, label):
        return tf.not_equal(label, -1)

    dataset = dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.filter(filter_invalid_images)
    dataset = dataset.shuffle(buffer_size=capacity).batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset

# # test get_batch
# import matplotlib.pyplot as plt
# BATCH_SIZE = 2
# CAPACITY = 256
# IMG_W = 350
# IMG_H = 350
#
# dataset = get_batch(imgs, label, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)
#
# # 创建一个迭代器来遍历数据集
# iterator = iter(dataset)
#
# # 获取一个批次的图像和标签
# images_batch, labels_batch = next(iterator)
#
# # 遍历一个批次的图像和标签，并显示每个图像及其对应的标签
# for j in range(BATCH_SIZE):
#     print('label: %d' % labels_batch[j].numpy())
#     plt.imshow(images_batch[j].numpy())
#     plt.show()