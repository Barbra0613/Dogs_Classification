# -*- coding: utf-8 -*-
# @Time : 2020/12/20
# @Author : Barbra
# @File : models.py
# @Software: PyCharm

import tensorflow as tf
from tensorflow import keras
import mobilenet_v3_small

def my_mobilenet_v3():
    # 构建模型
    model = tf.keras.Sequential([
        # 输入层，shape为(None,224,224,3)
        tf.keras.layers.Input((224, 224, 3)),
        # 输入到mobileNetV3中
        mobilenet_v3_small.MobileNetV3Small(),
        # # 将mobileNetV3的输出展平
        tf.keras.layers.Flatten(),
    ])

    return model


if __name__ == '__main__':
    model = my_mobilenet_v3()
    model.summary()
