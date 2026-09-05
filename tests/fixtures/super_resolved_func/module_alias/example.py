import tensorflow as tf
class CustomLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def get_config(self):
        return super().get_config()
