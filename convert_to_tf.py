import tensorflow as tf


def convert_to_tf(model_path, input_shape, output_path):
    # Convert the model to tf
    converter = tf.lite.TFLiteConverter.from_frozen_graph(model_path, #TensorFlow freezegraph .pb model file
                                                      input_arrays=['input'], # name of input arrays as defined in torch.onnx.export function before.
                                                      output_arrays=['output']  # name of output arrays defined in torch.onnx.export function before.
                                                      )
    
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    tflite_model = converter.convert()

    # Save the model
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
