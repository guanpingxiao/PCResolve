from tensorflow.keras.layers import Layer
class CustomLayer(Layer):
    def get_config(self):
        def nested():
            return super().get_config()
        return nested()
