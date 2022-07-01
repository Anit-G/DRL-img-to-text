from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras import layers, models

# define base model
states = [150,150,3]
actions = 27
base_model = Xception(weights="imagenet", include_top=False, input_shape=states)
base_model.trainable = False ## Not trainable weights


flatten_layer = layers.Flatten()
dense_layer_1 = layers.Dense(30, activation='relu')
dense_layer_2 = layers.Dense(30, activation='relu')
prediction_layer = layers.Dense(actions, activation='softmax')


model = models.Sequential([
    base_model,
    flatten_layer,
    dense_layer_1,
    dense_layer_2,
    prediction_layer
])

model.summary()