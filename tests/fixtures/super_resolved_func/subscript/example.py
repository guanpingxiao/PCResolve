from tensorflow.keras.layers import Layer, Other
class CustomLayer(Layer[int]):
    def get_config(self):
        return super().get_config()
