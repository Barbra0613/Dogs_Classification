# -*- coding: utf-8 -*-
# @Time : 2020/12/20
# @File : setting.py
# @Software : PyCharm
# @Desc : 配置参数


# 爬虫
# 图片类别和搜索关键词的映射关系
IMAGE_CLASS_KEYWORD_MAP = {
    'husky': '哈士奇',
    'shiba Inu': '柴犬',
    'samoyed': '萨摩耶',
    'Chow chow': '松狮',
    'Golden Retriever': '金毛',
}
# 图片保存根目录
IMAGES_ROOT = './images'
# 爬虫每个类别下载多少页图片
SPIDER_DOWNLOAD_PAGES = 50

# 数据
# 每个类别选取的图片数量（最大值）
SAMPLES_PER_CLASS = 800
# 参与训练的类别
CLASSES = ['husky','shiba Inu','samoyed','Chow chow','Golden Retriever']
# 参与训练的类别数量
CLASS_NUM = len(CLASSES)
# 类别->编号的映射
CLASS_CODE_MAP = {
   'husky': 0,
   'shiba Inu':1,
   'samoyed':2,
   'Chow chow':3,
   'Golden Retriever':4,
}
# 编号->类别的映射
CODE_CLASS_MAP = {
   0: '哈士奇',
   1: '柴犬',
   2: '萨摩耶',
   3: '松狮',
   4: '金毛',
}

# 随机数种子
RANDOM_SEED = 9
# 训练集比例
TRAIN_DATASET = 0.7
# 测试集比例
TEST_DATASET = 0.3
# mini_batch大小
BATCH_SIZE = 16

# 训练
# 学习率
LEARNING_RATE = 0.001
# 训练epoch数
TRAIN_EPOCHS = 10
# 保存训练模型的路径
MODEL_PATH = "checkpoint/mobilenetv3.ckpt"


