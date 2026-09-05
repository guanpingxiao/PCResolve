from tensorflow.keras.layers import Layer, Other
@decorate
class CustomLayer(Layer):
    def get_config(self):
        return super().get_config()
