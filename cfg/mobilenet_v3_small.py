    # -*- coding: utf-8 -*-
# @Time : 2020/12/20
# @File : mobilenet_v3_small.py
# @Software : PyCharm
# @Desc : MobileNetV3Small模型搭建

import tensorflow as tf
from cfg.mobilenet_v3_block import BottleNeck, h_swish
from cfg import settings

#定义MobileNetV3 Small版本的网络结构
class MobileNetV3Small(tf.keras.Model):
    def __init__(self):
        super(MobileNetV3Small, self).__init__()
        #卷积层
        self.conv1 = tf.keras.layers.Conv2D(filters=16,#滤波器的数量，即卷积层输出空间的维度
                                            kernel_size=(3, 3),#卷积核尺寸
                                            strides=2,#步长
                                            padding="same")#填充方式
        self.bn1 = tf.keras.layers.BatchNormalization()#批标准化
        #逐层定义Bneck块
        self.bneck1 = BottleNeck(in_size=16, exp_size=16, out_size=16, s=2, is_se_existing=True, NL="RE", k=3)
        self.bneck2 = BottleNeck(in_size=16, exp_size=72, out_size=24, s=2, is_se_existing=False, NL="RE", k=3)
        self.bneck3 = BottleNeck(in_size=24, exp_size=88, out_size=24, s=1, is_se_existing=False, NL="RE", k=3)
        self.bneck4 = BottleNeck(in_size=24, exp_size=96, out_size=40, s=2, is_se_existing=True, NL="HS", k=5)
        self.bneck5 = BottleNeck(in_size=40, exp_size=240, out_size=40, s=1, is_se_existing=True, NL="HS", k=5)
        self.bneck6 = BottleNeck(in_size=40, exp_size=240, out_size=40, s=1, is_se_existing=True, NL="HS", k=5)
        self.bneck7 = BottleNeck(in_size=40, exp_size=120, out_size=48, s=1, is_se_existing=True, NL="HS", k=5)
        self.bneck8 = BottleNeck(in_size=48, exp_size=144, out_size=48, s=1, is_se_existing=True, NL="HS", k=5)
        self.bneck9 = BottleNeck(in_size=48, exp_size=288, out_size=96, s=2, is_se_existing=True, NL="HS", k=5)
        self.bneck10 = BottleNeck(in_size=96, exp_size=576, out_size=96, s=1, is_se_existing=True, NL="HS", k=5)
        self.bneck11 = BottleNeck(in_size=96, exp_size=576, out_size=96, s=1, is_se_existing=True, NL="HS", k=5)
        #卷积层
        self.conv2 = tf.keras.layers.Conv2D(filters=576,
                                            kernel_size=(1, 1),
                                            strides=1,
                                            padding="same")
        self.bn2 = tf.keras.layers.BatchNormalization()#批标准化层
        self.avgpool = tf.keras.layers.AveragePooling2D(pool_size=(7, 7),strides=1)#平均池化层
        #卷积层
        self.conv3 = tf.keras.layers.Conv2D(filters=1280,
                                            kernel_size=(1, 1),
                                            strides=1,
                                            padding="same")
        self.conv4 = tf.keras.layers.Conv2D(filters=settings.CLASS_NUM,#输出维度等于类别数
                                            kernel_size=(1, 1),
                                            strides=1,
                                            padding="same",
                                            activation=tf.keras.activations.softmax)


    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = h_swish(x)

        x = self.bneck1(x, training=training)
        x = self.bneck2(x, training=training)
        x = self.bneck3(x, training=training)
        x = self.bneck4(x, training=training)
        x = self.bneck5(x, training=training)
        x = self.bneck6(x, training=training)
        x = self.bneck7(x, training=training)
        x = self.bneck8(x, training=training)
        x = self.bneck9(x, training=training)
        x = self.bneck10(x, training=training)
        x = self.bneck11(x, training=training)

        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = h_swish(x)
        x = self.avgpool(x)
        x = self.conv3(x)
        x = h_swish(x)
        x = self.conv4(x)


        return x


if __name__ == '__main__':
    model = MobileNetV3Small()
    model.summary()

