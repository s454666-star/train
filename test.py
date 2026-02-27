import os
import numpy as np
import tensorflow as tf
from PIL import Image
from dotenv import load_dotenv
from tensorflow.keras.losses import MeanSquaredError

# 註冊 mse 作為自定義物件
mse = MeanSquaredError()

def test_model():
    load_dotenv()

    model_path = "C:/Users/User/Pictures/train/models/portrait_model.h5"

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"找不到模型文件：{model_path}")

    print("正在加載模型...")
    model = tf.keras.models.load_model(model_path, custom_objects={"mse": mse})
    print("模型加載成功。")
    print("模型輸入形狀：", model.input_shape)

    # 動態調整輸入數據
    if len(model.input_shape) == 2:  # 生成器模型
        latent_dim = model.input_shape[1]
        input_data = np.random.normal(0, 1, (1, latent_dim))
    elif len(model.input_shape) == 4:  # 圖像處理模型
        input_shape = model.input_shape[1:]
        input_data = np.random.rand(*input_shape).astype(np.float32)
        input_data = np.expand_dims(input_data, axis=0)
    else:
        raise ValueError("不支援的模型輸入形狀！")

    print("正在生成圖像...")
    generated_image = model.predict(input_data)[0]

    # 圖像後處理
    generated_image = (generated_image + 1) / 2.0  # 轉換到 [0, 1]
    generated_image = np.clip(generated_image, 0, 1)
    generated_image = (generated_image * 255).astype(np.uint8)

    # 處理灰度圖或 RGB 圖
    if generated_image.shape[-1] == 1:
        generated_image = generated_image.squeeze(-1)
        image = Image.fromarray(generated_image, mode='L')
    else:
        image = Image.fromarray(generated_image)

    output_path = "test_generated_image.png"
    image.save(output_path)
    print(f"圖像已生成並保存為 {output_path}")

if __name__ == "__main__":
    test_model()
