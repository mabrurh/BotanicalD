import tensorflow as tf

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print("TensorFlow Version: ", tf.__version__)
print("CUDA Version: ", tf.sysconfig.get_build_info()["cuda_version"])
print("cuDNN Version: ", tf.sysconfig.get_build_info()["cudnn_version"])
