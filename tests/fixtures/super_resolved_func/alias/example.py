from tensorflow.keras.layers import Layer as KLayer
class CustomLayer(KLayer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def get_config(self):
        return super().get_config()
