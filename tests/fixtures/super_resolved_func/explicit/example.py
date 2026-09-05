from tensorflow.keras.layers import Layer, Other
class CustomLayer(Layer):
    def get_config(self):
        return super(CustomLayer, self).get_config()
