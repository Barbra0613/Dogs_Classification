# -*- coding: utf-8 -*-
# @Time : 2020/12/20
# @File : train.py
# @Software : PyCharm
# @Desc : 训练

import os
import tensorflow as tf
import models
from cfg import settings
from matplotlib import pyplot as plt
from data.data import train_db, test_db

# 从models文件中导入模型
model = models.my_mobilenet_v3()
model.summary()

#设置指数衰减学习率
exponential_decay = tf.keras.optimizers.schedules.ExponentialDecay(
                        initial_learning_rate=0.01, decay_steps=1, decay_rate=0.98)

# 配置优化器、损失函数、以及监控指标
model.compile(tf.keras.optimizers.Adam(exponential_decay),
              loss=tf.keras.losses.categorical_crossentropy,
              metrics=['accuracy'])

# 在每个epoch结束后尝试保存模型参数，设置断点续训
if os.path.exists(settings.MODEL_PATH + '.index'):
    print('-------------load the model-----------------')
    model.load_weights(settings.MODEL_PATH)

mb_callback = tf.keras.callbacks.ModelCheckpoint(filepath=settings.MODEL_PATH,
                                                 save_weights_only=True,
                                                 save_best_only=True)

#执行训练过程
history = model.fit(train_db, batch_size=settings.BATCH_SIZE, epochs=settings.TRAIN_EPOCHS, validation_data=(test_db), validation_freq=1,
                    callbacks=[mb_callback])

# 显示训练集和测试集的acc和loss曲线
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

# 绘制 Accuracy 曲线
plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

# 绘制 loss 曲线
plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()