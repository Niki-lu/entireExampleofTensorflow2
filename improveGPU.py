# -*-coding:utf-8-*-

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers, Sequential, regularizers


# 定义一个3x3卷积
def regularized_padded_conv(*args, **kwargs):
    return layers.Conv2D(*args, **kwargs, padding='same', kernel_regularizer=regularizers.l2(5e-5),
                         use_bias=False, kernel_initializer='glorot_normal')

# 定义 Basic Block 模块。对应Resnet18和Resnet34
class BasecBlock(layers.Layer):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasecBlock, self).__init__()
        # 1
        self.conv1 = regularized_padded_conv(out_channels, kernel_size=3, strides=stride)
        self.bn1 = layers.BatchNormalization()

        # 2
        self.conv2 = regularized_padded_conv(out_channels, kernel_size=3, strides=1)
        self.bn2 = layers.BatchNormalization()

        # 3
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = Sequential(
                [regularized_padded_conv(self.expansion * out_channels, kernel_size=3, strides=stride),
                 layers.BatchNormalization()])
        else:
            self.shortcut = lambda x, _: x

    @tf.function
    def call(self, inputs, training=False):

        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2(x)
        x = self.bn2(x, training=training)

        x_short = self.shortcut(inputs, training)

        x = x + x_short
        out = tf.nn.relu(x)

        return out

    # 定义 Bottleneck 模块。对应Resnet50,Resnet101和Resnet152


class Bottleneck(tf.keras.Model):
    expansion = 4

    def __init__(self, in_channels, out_channels, strides=1):
        super(Bottleneck, self).__init__()

        # 1
        self.conv1 = layers.Conv2D(out_channels, 1, 1, use_bias=False)
        self.bn1 = layers.BatchNormalization()

        # 2
        self.conv2 = layers.Conv2D(out_channels, 3, strides, padding='same', use_bias=False)
        self.bn2 = layers.BatchNormalization()
        self.conv3 = layers.Conv2D(out_channels, 1, 1, use_bias=False)
        self.bn3 = layers.BatchNormalization()

        if strides != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = selfSequential(
                [layers.Conv2D(self.expansion * out_channels, kernel_size=1, strides=strides, use_bias=False),
                 layers.BatchNormalization()])
        else:
            self.shortcut = lambda x, _: x

    @tf.function
    def call(self, inputs, training=False):

        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = tf.nn.rele(x)

        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = tf.nn.rele(x)

        x = self.conv3(x)
        x = self.bn3(x, training=training)

        x_short = self.shortcut(x, training=training)
        x = x + x_short
        out = tf.nn.relu(x)
        return out
    # 自定义模型，ResBlock 模块。继承keras.Model或者keras.Layer都可以


class ResNet(tf.keras.Model):
    # 第1个参数blocks：对应2个自定义模块BasicBlock和Bottleneck, 其中BasicBlock对应res18和res34，Bottleneck对应res50,res101和res152，
    # 第2个参数layer_dims：[2, 2, 2, 2] 4个Res Block，每个包含2个Basic Block
    # 第3个参数num_classes：我们的全连接输出，取决于输出有多少类
    def __init__(self, blocks, layer_dims, initial_filters=64, num_classes=100):
        super(ResNet, self).__init__()

        self.in_channels = initial_filters
        #
        self.stem = Sequential([regularized_padded_conv(initial_filters, kernel_size=3, strides=1),
                                layers.BatchNormalization()])
        #
        self.layer1 = self.build_resblock(blocks, initial_filters, layer_dims[0], stride=1)
        self.layer2 = self.build_resblock(blocks, initial_filters * 2, layer_dims[1], stride=2)
        self.layer3 = self.build_resblock(blocks, initial_filters * 4, layer_dims[2], stride=2)
        self.layer4 = self.build_resblock(blocks, initial_filters * 8, layer_dims[3], stride=2)

        self.final_bn = layers.BatchNormalization()

        self.avg_pool = layers.GlobalAveragePooling2D()
        self.dense = layers.Dense(num_classes)

    def build_resblock(self, blocks, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        res_block = Sequential()

        for stride in strides:
            res_block.add(blocks(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return res_block

    @tf.function
    def call(self, inputs, training):
        x = self.stem(inputs, training)
        x = tf.nn.relu(x)

        x = self.layer1(x, training=training)
        x = self.layer2(x, training=training)
        x = self.layer3(x, training=training)
        x = self.layer4(x, training=training)

        x = self.final_bn(x, training=training)
        x = tf.nn.relu(x)
        x = self.avg_pool(x)
        out = self.dense(x)

        return out


def resnet18():
    return ResNet(BasecBlock, [2, 2, 2, 2])


def resnet34():
    return ResNet(BasecBlock, [3, 4, 6, 3])


def resnet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def resnet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def resnet151():
    return ResNet(Bottleneck, [3, 8, 36, 3])


import pandas as pd
import numpy as np
import os
import tensorflow as tf

from tensorflow.keras import layers, optimizers, datasets, Sequential

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.random.set_seed(2345)
penalty_parameter = 1
batchs = 300

# 数据normalize
# 下面这2个值均值和方差，怎么得到的。其实是统计所有imagenet的图片(几百万张)的均值和方差；
# 所有者2个数据比较有意义，因为本质上所有图片的分布都和imagenet图片的分布基本一致。
# 这6个数据基本是通用的，网上一搜就能查到
img_mean = tf.constant([0.4914, 0.4822, 0.4465])
img_std = tf.constant([0.2023, 0.1994, 0.2010])


# 打印时间分割线
@tf.function
def printbar():
    ts = tf.timestamp()
    today_ts = ts % (24 * 60 * 60)

    hour = tf.cast(today_ts // 3600 + 8, tf.int32) % tf.constant(24)
    minite = tf.cast((today_ts % 3600) // 60, tf.int32)
    second = tf.cast(tf.floor(today_ts % 60), tf.int32)

    def timeformat(m):
        if tf.strings.length(tf.strings.format("{}", m)) == 1:
            return (tf.strings.format("0{}", m))
        else:
            return (tf.strings.format("{}", m))

    timestring = tf.strings.join([timeformat(hour), timeformat(minite),
                                  timeformat(second)], separator=":")
    tf.print("==========" * 6, end="")
    tf.print(timestring)


def normalize(x, mean=img_mean, std=img_std):
    # x shape: [224, 224, 3]
    # mean：shape为1；这里用到了广播机制。我们安装好右边对齐的原则，可以得到如下；
    # mean : [1, 1, 3], std: [3]        先插入1
    # mean : [224, 224, 3], std: [3]    再变为224
    x = (x - mean) / std
    return x


# 数据预处理.[-1~1]
def preprocess(x, y):
    x = tf.image.random_flip_left_right(x)
    # x: [0,255]=> 0~1 或者-0.5~0.5   其次：normalizaion
    x = tf.cast(x, dtype=tf.float32) / 255.
    # 0~1 => D(0,1) 调用函数；
    x = normalize(x)
    y = tf.cast(y, dtype=tf.int32)
    return x, y


# 数据集的加载
(x, y), (x_test, y_test) = datasets.cifar100.load_data()

# 或者tf.squeeze(y, axis=1)把1维度的squeeze掉。
y = tf.squeeze(y)
y_test = tf.squeeze(y_test)
print(x.shape, y.shape, x_test.shape, y_test.shape)

ds_train = tf.data.Dataset.from_tensor_slices((x, y)).shuffle(5000).batch(batchs) \
    .map(preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE).prefetch(tf.data.experimental.AUTOTUNE)
ds_test = tf.data.Dataset.from_tensor_slices((x_test, y_test)).shuffle(5000).batch(batchs) \
    .map(preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE).prefetch(tf.data.experimental.AUTOTUNE)

# 样本的形状
sample = next(iter(ds_train))
print('sample', sample[0].shape, sample[1].shape, tf.reduce_min(sample[0]), tf.reduce_max(sample[0]))

# @tf.function
model = resnet18()
model.build(input_shape=(None, 32, 32, 3))
model.summary()

# 学习率及损失函数
lr = 6e-4
optimizer = optimizers.Adam(lr=lr)


# 模型训练，求导及参数更新
@tf.function
def my_gradient(x, y, lr):
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        y_onehot = tf.one_hot(y, depth=100)
        loss = tf.losses.categorical_crossentropy(y_onehot, logits, from_logits=True)
        loss = tf.reduce_mean(loss)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss


# 模型循环训练
printbar()
t0 = tf.timestamp()
for epoch in tf.range(5):
    t1 = tf.timestamp()
    for step, (x, y) in enumerate(ds_train):
        """
        @tf.function
        with tf.GradientTape() as tape:
            logits=model(x,training=True)
            #损失函数使用categorical_crossentropy时需要将y进行onehot变化，如不进行onehot可以使用sparse_categorical_crossentropy
            y_onehot=tf.one_hot(y,depth=100)
            #from_logits=True时，将Softmax和交叉熵同时实现
            loss=tf.losses.categorical_crossentropy(y_onehot,logits,from_logits=True)
            loss=tf.reduce_mean(loss)
        #梯度求解
        #grads=tape.gradient(loss,model.trainable_variables)
        grads=my_gradient(x,y)
        #梯度更新
        optimizer.apply_gradients(zip(grads,model.trainable_variables))
        """
        # 模型训练，求导及更新参数
        loss = my_gradient(x, y, lr)
        # 调整学习率
        if epoch <= 30:
            lr = 6e-4
        elif epoch > 30 and epoch <= 80:
            lr = 3e-4
        elif epoch > 80 and epoch <= 150:
            lr = 1e-4
        elif epoch > 150 and epoch <= 500:
            lr = 3e-5
        else:
            lr = 1e-6

        if step % 100 == 0:
            tf.print(epoch, step, 'loss:', float(loss), lr)

    # 评估测试集
    total_num = 0
    total_correct = 0
    for x, y in ds_test:
        output = model(x, training=True)
        # 进行softmax计算
        prob = tf.nn.softmax(output, axis=1)
        # 进行标签提取，提取概率最大的index
        pred = tf.argmax(prob, axis=1)
        # 类型变换
        pred = tf.cast(pred, dtype=tf.int32)
        # 将预测值和真实标签比较
        correct = tf.cast(tf.equal(pred, y), dtype=tf.int32)
        correct = tf.reduce_sum(correct)
        total_num += x.shape[0]
        total_correct += int(correct)

    acc = total_correct / total_num
    print(epoch, 'test_acc:', acc, 'use time per epoch(s/epoch)', round((tf.timestamp() - t1).numpy(), 4))
printbar()
print("end training,use time(h):", round((tf.timestamp() - t0).numpy() / (60 * 60), 2))
