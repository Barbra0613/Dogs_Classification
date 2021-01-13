# -*- coding: utf-8 -*-
# @Time : 2020/12/20
# @File : eval.py
# @Software : PyCharm
# @Desc : 模型验证


import tensorflow as tf
from pandas import np

import os
from data.data import test_db
from models import my_mobilenet_v3
from cfg import settings
from keras import backend as K

# 创建模型
model = my_mobilenet_v3()
# 加载参数
if os.path.exists(settings.MODEL_PATH+ '.index'):
    print('-------------load the model-----------------')
    model.load_weights(settings.MODEL_PATH)
else:
    print('false')
# 编译模型
model.compile(tf.keras.optimizers.Adam(settings.LEARNING_RATE),
              loss=tf.keras.losses.categorical_crossentropy,
              metrics=['accuracy'])
# 测试集accuracy
print('test', model.evaluate(test_db))

#定义Macro-F1计算函数
def f1(y_hat, y_true, THRESHOLD=0.5):
    '''
    y_hat是未经过sigmoid函数激活的
    输出的f1为Macro-F1
    '''
    epsilon = 1e-7
    y_hat = y_hat > THRESHOLD
    y_hat = np.int8(y_hat)
    tp = np.sum(y_hat * y_true, axis=0)
    fp = np.sum(y_hat * (1 - y_true), axis=0)
    fn = np.sum((1 - y_hat) * y_true, axis=0)

    p = tp / (tp + fp + epsilon)  # epsilon的意义在于防止分母为0，否则当分母为0时python会报错
    r = tp / (tp + fn + epsilon)

    f1 = 2 * p * r / (p + r + epsilon)
    f1 = np.where(np.isnan(f1), np.zeros_like(f1), f1)

    return np.mean(f1)

# 计算Macro-F1
y_t = np.array([])
y_p = np.array([])
for x, y in test_db:

    y_pred = model(x)
    y_pred = tf.argmax(y_pred, axis=1).numpy()
    if  y_p.size==0 :
        y_p = y_pred
    else:
        y_p = np.concatenate((y_p,y_pred),axis=0)

    y_true = tf.argmax(y, axis=1).numpy()
    if  y_t.size==0 :
        y_t = y_true
    else:
        y_t = np.concatenate((y_t,y_true),axis=0)

mf = f1(y_pred, y_true)
print('F1 score:', mf)

# 查看识别错误的数据
for x, y in test_db:
    y_pred = model(x)
    y_pred = tf.argmax(y_pred, axis=1).numpy()
    y_true = tf.argmax(y, axis=1).numpy()

    batch_size = y_pred.shape[0]
    for i in range(batch_size):
        if y_pred[i] != y_true[i]:
            print('{} 被错误识别成 {}!'.format(settings.CODE_CLASS_MAP[y_true[i]], settings.CODE_CLASS_MAP[y_pred[i]]))