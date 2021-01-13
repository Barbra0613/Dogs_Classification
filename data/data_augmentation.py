# -*- coding: utf-8 -*-
# @Time : 2021/1/8
# @File : data_augmentation.py
# @Software : PyCharm
# @Desc :

import tensorflow as tf
from tensorflow.python.keras.preprocessing.image \
                        import ImageDataGenerator   , array_to_img, img_to_array, load_img
import os
from cfg import settings

#定义图片生成器
datagen = ImageDataGenerator(
    rotation_range=30,# 随机30度旋转
    width_shift_range=0.15,#宽度偏移
    height_shift_range=0.15,#高度偏移
    shear_range=0.2,#剪切强度
    zoom_range=0.2,#将图像随机缩放阈量20%
    horizontal_flip=True,# 水平翻转
    fill_mode='nearest')#使用近邻填充

cwd=os.getcwd()
dir=cwd+'\\images\\'
dir=settings.IMAGES_ROOT

index = 0#计数，控制只对数据集中的部分进行数据增强
for i, filename in enumerate(os.listdir(dir)):
    save_dir=dir+filename+'\\'#数据增强后，生成图片的保存路径
    for i,filename2 in enumerate(os.listdir(save_dir)):
        index+=1
        if index%2!=0:#只对数据集中部分数据进行数据增强，并将变换后的图像保存到原文件夹下
            continue
        i=0#计数，控制对每张图片使用数据增强的次数
        #每一个图片文件的路径
        filename2=save_dir+filename2
        img=load_img(filename2)
        x = img_to_array(img)#将图片数据转换为数组
        x = x.reshape((1,) + x.shape)  #将数据提升一个维度
        print(filename2,'is augmenting...')
        for batch in datagen.flow(x, batch_size=10,
                                  save_to_dir=save_dir, #保存路径，
                                  save_prefix='a', #图片前缀名
                                  save_format='jpeg'):#图片格式
            i += 1#计数，控制对每张图片使用数据增强的次数
            if i >= 1:#当前，对每张图片最多只执行一次数据增强的操作
                break
        print(filename,'has augmented')

