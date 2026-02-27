# test_flux_dev.py

import os
from dotenv import load_dotenv
import torch
from diffusers import FluxPipeline
from PIL import Image

def main():
    # 加載 .env 文件中的環境變量
    load_dotenv()
    huggingface_token = os.getenv("HUGGINGFACE_TOKEN")

    if not huggingface_token:
        raise ValueError("請在 .env 文件中設置 HUGGINGFACE_TOKEN。")

    # 設置設備 (GPU 如果可用，否則使用 CPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用設備: {device}")

    # 加載 FluxPipeline
    print("加載 FLUX.1 [dev] 模型...")
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        use_auth_token=huggingface_token
    )

    if device == "cpu":
        pipe.enable_model_cpu_offload()  # 在使用 CPU 時節省內存

    pipe = pipe.to(device)

    # 設置提示詞
    prompt = "Cute, naked, no clothes, high school student, student, junior high school student, teenager, classroom, small breasts, A cup, beautiful girl, real person, nipples"

    print(f"生成圖像，提示詞: '{prompt}'")
    # 生成圖像
    with torch.autocast(device):
        output = pipe(
            prompt,
            height=1024,
            width=1024,
            guidance_scale=7.5,
            num_inference_steps=30,
            max_sequence_length=512,
            generator=torch.Generator(device).manual_seed(0)
        )

    image: Image.Image = output.images[0]
    image_path = "flux_dev_output.png"
    image.save(image_path)
    print(f"圖像已保存至 {image_path}")

if __name__ == "__main__":
    main()
