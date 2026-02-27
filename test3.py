import tensorflow as tf

print("可用的設備列表:")
print(tf.config.list_logical_devices())

try:
    cuda_version = tf.sysconfig.get_build_info().get("cuda_version", "未檢測到 CUDA")
    cudnn_version = tf.sysconfig.get_build_info().get("cudnn_version", "未檢測到 cuDNN")
    print("已加載的 CUDA 庫版本:", cuda_version)
    print("已加載的 cuDNN 庫版本:", cudnn_version)
except KeyError as e:
    print(f"無法獲取 CUDA 或 cuDNN 信息: {e}")
