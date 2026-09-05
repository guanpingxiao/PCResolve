from tensorflow.keras.layers import Layer
class CustomLayer(Layer):
    def get_config(self, super):
        return super().get_config()
