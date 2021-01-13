# -*- coding: utf-8 -*-
# @Time : 2020/12/20
# @File : mobilenet_v3_block.py
# @Software : PyCharm
# @Desc : MobileNetV3Small模型搭建

import tensorflow as tf


def h_sigmoid(x):
    return tf.nn.relu6(x + 3) / 6
def h_swish(x):
    return x * h_sigmoid(x)

class SEBlock(tf.keras.layers.Layer):
    def __init__(self, input_channels, r=16):
        super(SEBlock, self).__init__()
        self.pool = tf.keras.layers.GlobalAveragePooling2D()
        self.fc1 = tf.keras.layers.Dense(units=input_channels // r)
        self.fc2 = tf.keras.layers.Dense(units=input_channels)

    def call(self, inputs, **kwargs):
        branch = self.pool(inputs)
        branch = self.fc1(branch)
        branch = tf.nn.relu(branch)
        branch = self.fc2(branch)
        branch = h_sigmoid(branch)
        branch = tf.expand_dims(input=branch, axis=1)
        branch = tf.expand_dims(input=branch, axis=1)
        output = inputs * branch
        return output

#MobileNetV2的BottleNeck模块
class BottleNeck(tf.keras.layers.Layer):
    def __init__(self, in_size, exp_size, out_size, s, is_se_existing, NL, k):
        super(BottleNeck, self).__init__()
        self.stride = s#步长
        self.in_size = in_size#输入图片尺寸
        self.out_size = out_size#输出图片尺寸
        # 是否使用MobileNetV3的Squeeze and excite轻型注意力机制
        self.is_se_existing = is_se_existing
        self.NL = NL#选择使用relu或hard_swish激活函数
        #卷积层1
        self.conv1 = tf.keras.layers.Conv2D(filters=exp_size,#滤波器的数量，即卷积层输出空间的维度
                                            kernel_size=(1, 1),#卷积核大小
                                            strides=1,#步长
                                            padding="same")#填充方式
        self.bn1 = tf.keras.layers.BatchNormalization()#批标准化
        #深度可分离卷积，
        self.dwconv = tf.keras.layers.DepthwiseConv2D(kernel_size=(k, k),
                                                      strides=s,
                                                      padding="same")
        self.bn2 = tf.keras.layers.BatchNormalization()#批标准化
        self.se = SEBlock(input_channels=exp_size)#Squeeze and excite轻型注意力机制
        #卷据层2
        self.conv2 = tf.keras.layers.Conv2D(filters=out_size,
                                            kernel_size=(1, 1),
                                            strides=1,
                                            padding="same")
        self.bn3 = tf.keras.layers.BatchNormalization()#批标准化
        self.linear = tf.keras.layers.Activation(tf.keras.activations.linear)#线性激活函数

    def call(self, inputs, training=None, **kwargs):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        if self.NL == "HS":
            x = h_swish(x)
        elif self.NL == "RE":
            x = tf.nn.relu6(x)
        x = self.dwconv(x)
        x = self.bn2(x, training=training)
        if self.NL == "HS":
            x = h_swish(x)
        elif self.NL == "RE":
            x = tf.nn.relu6(x)
        if self.is_se_existing:
            x = self.se(x)
        x = self.conv2(x)
        x = self.bn3(x, training=training)
        x = self.linear(x)

        if self.stride == 1 and self.in_size == self.out_size:
            x = tf.keras.layers.add([x, inputs])

        return x
