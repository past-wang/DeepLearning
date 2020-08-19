import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv('../DeepLearning/usual datasets/Income1.csv')
plt.scatter(data.Education,data.Income)
x = data.Education
y = data.Income

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(1,input_shape=(1,)))#Dense(x，),维度x,     添加层
model.summary()
model.compile(optimizer='adam',
             loss='mse'
)


history = model.fit(x,y,epochs=5000)
model.predict(x)
model.predict(y)