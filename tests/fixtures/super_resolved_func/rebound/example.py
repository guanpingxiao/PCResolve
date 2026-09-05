from tensorflow.keras.layers import Layer
class CustomLayer(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def get_config(self):
        return super().get_config()
from otherlib import Layer
