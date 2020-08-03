import tensorflow as tf
import numpy as np

model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[1])
])
model.compile(optimizer='sgd', loss = 'mean_squared_error')

x = np.array([-1,0,1,2,3,4], dtype = float)
y = np.array([-3,-1,1,3,5,7], dtype = float)

model.fit(x,y, epochs=300)

print(model.predict([5]))
