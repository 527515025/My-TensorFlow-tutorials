import tensorflow as tf
from tensorflow.keras import layers, models
# 结构
# conv1   卷积层 1
# pooling1_lrn  池化层 1
# conv2  卷积层 2
# pooling2_lrn 池化层 2
# local3 全连接层 1
# local4 全连接层 2
# softmax 全连接层 3

def inference(input_shape, n_classes):
    model = models.Sequential()

    #修改
    model.add(layers.Input(shape=input_shape))  # 使用 Input 层定义输入形状

    # Conv1，第一个卷积层，使用3x3的卷积核，输出16个特征图，使用ReLU激活函数
    model.add(layers.Conv2D(16, (3, 3), activation='relu', padding='same', name='conv1'))

    # Pooling1_lrn，添加一个最大池化层，使用3x3的池化窗口，步幅为2x2，然后进行批量归一化
    model.add(layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='pooling1'))
    model.add(layers.BatchNormalization(name='norm1'))

    # Conv2，第二个卷积层，使用3x3的卷积核，输出16个特征图，使用ReLU激活函数
    model.add(layers.Conv2D(16, (3, 3), activation='relu', padding='same', name='conv2'))

    # Pooling2_lrn，进行批量归一化，然后添加一个最大池化层，使用3x3的池化窗口，步幅为1x1
    model.add(layers.BatchNormalization(name='norm2'))
    model.add(layers.MaxPooling2D((3, 3), strides=(1, 1), padding='same', name='pooling2'))

    # Flatten，将多维输入一维化，为全连接层做准备
    model.add(layers.Flatten())

    # Local3，第一个全连接层，有128个神经元，使用ReLU激活函数
    model.add(layers.Dense(128, activation='relu', name='local3'))

    # Local4，第二个全连接层，有128个神经元，使用ReLU激活函数
    model.add(layers.Dense(128, activation='relu', name='local4'))

    # Softmax，输出层，有n_classes个神经元，使用softmax激活函数
    model.add(layers.Dense(n_classes, activation='softmax', name='softmax_linear'))

    return model


# 计算模型的损失，SparseCategoricalCrossentropy是一个用于多分类问题的损失函数，适用于标签是整数的情况
def losses(logits, labels):
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    loss = loss_fn(labels, logits)
    return loss


# 定义模型的训练过程
def trainning(model, loss, learning_rate):
    # 使用Adam优化器，learning_rate是学习率
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    # compile: 编译模型，指定优化器、损失函数和评估指标（准确率）
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    return model


# 评估模型的性能
def evaluation(model, images, labels):
    loss, accuracy = model.evaluate(images, labels)
    return accuracy
