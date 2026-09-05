from tensorflow.keras.layers import Layer, Other
class Parent(Layer):
    pass
class CustomLayer(Parent):
    def get_config(self):
        return super().get_config()
