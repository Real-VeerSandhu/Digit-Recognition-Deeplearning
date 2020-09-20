import tensorflow as tf
import numpy as np
from PIL import Image

base_model = tf.keras.models.load_model('models/seq_model2.h5')
prob_model = tf.keras.Sequential([base_model, tf.keras.layers.Softmax()])

img1 = Image.open("new digits/7.png").convert(mode="F")
array1 = np.array(img1.getdata())

x = array1.reshape(28,28)
x = np.expand_dims(x, axis=0) / 255.0

prob_result = prob_model.predict(x)

print(prob_result)
print(np.argmax(prob_result))