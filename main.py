from keras.models import load_model
import tensorflow as tf

model = load_model("models\seq_model1.h5")
prob_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])