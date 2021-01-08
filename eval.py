# -*- coding: utf-8 -*-
# @Time : 2020/12/20
# @File : eval.py
# @Software : PyCharm
# @Desc : 模型验证

import tensorflow as tf
from data import test_db
from models import my_mobilenet_v3
import settings

# 创建模型
model = my_mobilenet_v3()
# 加载参数
model.load_weights(settings.MODEL_PATH)
# 编译模型
model.compile(tf.keras.optimizers.Adam(settings.LEARNING_RATE), loss=tf.keras.losses.categorical_crossentropy,
              metrics=['accuracy'])
# 测试集accuracy
print('test', model.evaluate(test_db))

# 查看识别错误的数据
for x, y in test_db:
    y_pred = model(x)
    y_pred = tf.argmax(y_pred, axis=1).numpy()
    y_true = tf.argmax(y, axis=1).numpy()
    batch_size = y_pred.shape[0]
    for i in range(batch_size):
        if y_pred[i] != y_true[i]:
            print('{} 被错误识别成 {}!'.format(settings.CODE_CLASS_MAP[y_true[i]], settings.CODE_CLASS_MAP[y_pred[i]]))
