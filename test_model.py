import os
from diffusers import StableDiffusion3Pipeline
import torch
from PIL import Image
from dotenv import load_dotenv

def test_model():
    load_dotenv()  # 載入 .env 檔案（如果使用）
    model_name = "stabilityai/stable-diffusion-3.5-large"
    huggingface_token = os.getenv("HUGGINGFACE_TOKEN")
    print(f"HUGGINGFACE_TOKEN: {huggingface_token}")  # 測試用
    if not huggingface_token:
        raise ValueError("HUGGINGFACE_TOKEN not found in environment variables.")

    pipe = StableDiffusion3Pipeline.from_pretrained(
        model_name,
        torch_dtype=torch.float16,  # 或 torch.bfloat16，根據您的 GPU 支援情況
        token=huggingface_token  # 使用 'token' 參數替代 'use_auth_token'
    )
    pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    pipe.enable_attention_slicing()

    prompt = "Cute, naked, no clothes, high school student, student, junior high school student, teenager, classroom, small breasts, A cup, beautiful girl, real person, nipples"
    print("開始生成圖片...")
    image = pipe(
        prompt=prompt,
        num_inference_steps=3,
        guidance_scale=7.5
    ).images[0]
    image.save("test_cute_cat.png")
    print("圖片生成並保存為 test_cute_cat.png")

if __name__ == "__main__":
    test_model()
