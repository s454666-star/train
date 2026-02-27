import tensorflow as tf
import torch

print("TensorFlow 版本：", tf.__version__)
print("GPU 设备列表：", tf.config.list_physical_devices('GPU'))
print("PyTorch 版本：", torch.__version__)
print("CUDA 是否可用：", torch.cuda.is_available())
print("GPU 数量：", torch.cuda.device_count())