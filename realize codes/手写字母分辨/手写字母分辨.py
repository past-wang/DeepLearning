#1. 导包
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


#2. 确定训练集和测试集

#载入测试集
mnist=tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test)=mnist.load_data()
x_train,x_test=x_train/225.0,x_test/225.0

#Seauential搭建网络结构
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dense(10,activation='softmax')
])
#compile
model.compile(optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['sparse_categorical_accuracy']
    )
#fit
model.fit(x_train,y_train,batch_size=32,epochs=30,validation_data=(x_test,y_test),validation_freq=1)

model.summary()