# 加载需要用到的库
# 数据处理库
import numpy as np
import pandas as pd
# 深度学习库
import keras
import keras.models
# json
import json

# 读数据
data_train = pd.read_csv('./Data/train.csv')
data_test = pd.read_csv('./Data/test.csv')

# 提取数据和标签
y_train = data_train.iloc[:,0]
X_train = data_train.iloc[:,1:]
X_test = data_test

# 归一化
X_train = X_train / 255.0
X_test = X_test / 255.0

# 将数据变形为需要的维度
# 因为输入CNN的图片数据维度应该为（N,H,W,D）
X_train = X_train.values.reshape(-1,28,28,1)
X_test = X_test.values.reshape(-1,28,28,1)

# 将标签转换为独热码
y_train = keras.utils.to_categorical(y_train,num_classes=10)



# 从文件加载整个模型模型并预测
model = keras.models.load_model('model_all.h5')
loss,acc = model.evaluate(X_train, y_train, batch_size=64)
print("loss:",loss,"acc:",acc)



# 分开加载模型和权重并预测
# 加载模型
with open('model.json','r',encoding='utf-8') as json_file:
    model_jason = json.load(json_file)
model = keras.models.model_from_json(model_jason)
# 加载权重
model.load_weights('model_weights.h5')
# 分开加载时需要编译模型
#optimizer = keras.optimizers.SGD(lr=0.01,momentum=0.9,decay=1e-5,nesterov=True)
optimizer = keras.optimizers.RMSprop(lr=0.001, rho=0.9)
#optimizer = keras.optimizers.Adam(lr=0.001,beta_1=0.9,beta_2=0.999)
model.compile(optimizer=optimizer, loss='categorical_crossentropy',metrics=["accuracy"])
# 预测
loss,acc = model.evaluate(X_train, y_train, batch_size=64)
print("loss:",loss,"acc:",acc)