# -*- coding: utf-8 -*-
# @Time : 2021/1/8
# @Author : Barbra
# @File : data_augamentation.py 
# @Software : PyCharm
# @Desc :
# from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import tensorflow as tf
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import os

cwd=os.getcwd()

datagen = ImageDataGenerator(
    # samplewise_center=False,
    # featurewise_std_normalization=False,
    # samplewise_std_normalization=False,
    # zca_whitening=False,
    # zca_epsilon=1e-06,
    rotation_range=30,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.2,#剪切强度
    zoom_range=0.2,#随机放大
    horizontal_flip=True,
    fill_mode='nearest')


dir=cwd+'\\images'

index = 0#控制只对数据集中的部分进行数据增强

dir=dir+'\\'
save_dir=cwd+'\\images\\a'

for i, filename in enumerate(os.listdir(dir)):
    # print(filename)
    second_dir=dir+filename+'\\'
    for i,filename2 in enumerate(os.listdir(second_dir)):
        # print(filename2)
        index+=1
        if index%10!=0:#只对数据集中1/10的数据进行数据增强，并将变换后的图像保存到原文件夹下
            continue
        i=0
        filename2=second_dir+filename2
        # dir=dir+filename

        img=load_img(filename2)
        #     print(file_name)
        x = img_to_array(img)  # this is a Numpy array with shape (536, 536, 3)
        x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 536, 536, 3)
        print(filename2,'is augmenting...')
        for batch in datagen.flow(x, batch_size=10,  # save_to_dir 要保存的文件夹   prefix图片名字   format图片的格式
                                  save_to_dir=second_dir, save_prefix='a', save_format='jpeg'):
            i += 1
            if i >= 1:
                break
        print(filename,'has augmented')

