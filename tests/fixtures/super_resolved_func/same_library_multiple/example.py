from tensorflow.keras.layers import Layer, Dense
class CustomLayer(Layer, Dense):
    def get_config(self):
        return super().get_config()
