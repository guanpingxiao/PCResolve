from tensorflow.keras.layers import Layer
class CustomLayer(Layer):
    def get_config(self):
        return super().get_config()
from otherlib import Parent
class CustomLayer(Parent):
    def get_config(self):
        return super().get_config()
