# -*- coding: utf-8 -*-
# @Time : 2021/1/8
# @File : app.py 
# @Software : PyCharm
# @Desc :

import tensorflow as tf
from flask import Flask
from flask import jsonify
from flask import request, render_template

import os,sys
from cfg import settings

# 导入模型
from models import my_mobilenet_v3
app = Flask(__name__)

os.chdir(os.path.dirname(sys.argv[0]))
#加载模型
model=my_mobilenet_v3()
# 加载训练好的参数
if os.path.exists(settings.MODEL_PATH + '.index'):
    print('-------------load the model-----------------')
    model.load_weights(settings.MODEL_PATH)

@app.route('/', methods=['GET'])
#首页，vue入口
def index():
    """
    首页，vue入口
    """
    return render_template('index.html')

@app.route('/api/v1/dogs_classify/', methods=['POST'])
#宠物狗图片分类接口
def dogs_classify():
    
    img_str = request.files.get('file').read()
    # 进行数据预处理
    x = tf.image.decode_image(img_str, channels=3)
    x = tf.image.resize(x, (224, 224))
    x = x / 255.
    x = (x - tf.constant(settings.IMAGE_MEAN)) / tf.constant(settings.IMAGE_STD)
    x = tf.reshape(x, (1, 224, 224, 3))
    # 预测
    y_pred = model(x)
    dog_cls_code = tf.argmax(y_pred, axis=1).numpy()[0]
    dog_cls_prob = float(y_pred.numpy()[0][dog_cls_code])
    dog_cls_prob = '{}%'.format(int(dog_cls_prob * 100))
    dog_class = settings.CODE_CLASS_MAP.get(dog_cls_code)
    # 将预测结果组织成json
    res = {
        'code': 0,
        'data': {
            'dog_cls': dog_class,
            'probability': dog_cls_prob,
            'msg': '<br><br><strong style="font-size: 48px;">{}</strong> <span style="font-size: 24px;"'
                   '>概率<strong>{}</strong></span>'.format(dog_class, dog_cls_prob),
        }
    }
    # 返回json数据
    return jsonify(res)#


if __name__ == '__main__':#启动web应用
    app.run(host='0.0.0.0', port=settings.WEB_PORT)