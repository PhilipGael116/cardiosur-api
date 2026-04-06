import tensorflow as tf
import numpy as np

interpreter = tf.lite.Interpreter(model_path="model/arrythmia.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Input details:", input_details)
print("Output details:", output_details)