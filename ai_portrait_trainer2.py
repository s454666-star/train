# ai_portrait_trainer.py

import sys
import os
import mysql.connector
from mysql.connector import Error
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QVBoxLayout, QFileDialog,
    QProgressBar, QTextEdit, QLabel, QHBoxLayout, QSpinBox,
    QCheckBox, QListWidget, QMessageBox, QListWidgetItem, QDialog, QGroupBox, QRadioButton, QInputDialog
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt5.QtGui import QPixmap, QIcon
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
import json
import datetime
import torch
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
from dotenv import load_dotenv
from tensorflow.keras.optimizers import Adam, RMSprop
import time

# 資料庫工具
class Database:
    def __init__(self):
        load_dotenv()  # 載入 .env 檔案
        self.host = os.getenv('DB_HOST')
        self.user = os.getenv('DB_USER')
        self.password = os.getenv('DB_PASSWORD')
        self.database = os.getenv('DB_NAME')
        self.port = int(os.getenv('DB_PORT', 3306))
        self.connection = None
        self.connect()

    def connect(self):
        try:
            self.connection = mysql.connector.connect(
                host=self.host,
                user=self.user,
                password=self.password,
                database=self.database,
                port=self.port
            )
            if self.connection.is_connected():
                print("資料庫連接成功")
        except Error as e:
            print(f"資料庫連接錯誤: {e}")
            self.connection = None

    def reconnect(self):
        self.close()
        self.connect()

    def close(self):
        if self.connection and self.connection.is_connected():
            self.connection.close()
            print("資料庫連接已關閉")
            self.connection = None

    def log_training(self, image_path, model_version, loss, accuracy, validation_loss, validation_accuracy, training_time_seconds, trained_by, notes):
        if self.connection is None or not self.connection.is_connected():
            self.connect()
        if self.connection and self.connection.is_connected():
            try:
                cursor = self.connection.cursor()
                query = """
                INSERT INTO training_logs
                (image_path, model_version, loss, accuracy, validation_loss, validation_accuracy, training_time_seconds, trained_by, notes)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """
                cursor.execute(query, (
                    image_path,
                    model_version,
                    float(loss),  # 確保是 Python float
                    float(accuracy),  # 確保是 Python float
                    float(validation_loss),  # 確保是 Python float
                    float(validation_accuracy),  # 確保是 Python float
                    int(training_time_seconds),  # 確保是 int
                    str(trained_by),  # 確保是 string
                    str(notes)  # 確保是 string
                ))
                self.connection.commit()
                cursor.close()
                print(f"訓練圖片路徑及詳細紀錄已記錄: {image_path}")
            except Error as e:
                print(f"插入訓練紀錄錯誤: {e}")
                self.connection = None  # 斷開連接以便下次重新連接

    def log_generated_image(self, input_image, generated_image, parameters, text_prompt, generation_mode):
        if self.connection is None or not self.connection.is_connected():
            self.connect()
        if self.connection and self.connection.is_connected():
            try:
                cursor = self.connection.cursor()
                query = """
                INSERT INTO generated_images (input_image_path, generated_image_path, parameters, text_prompt, generation_mode)
                VALUES (%s, %s, %s, %s, %s)
                """
                cursor.execute(query, (
                    input_image,
                    generated_image,
                    json.dumps(parameters),  # 確保是字符串
                    text_prompt,
                    generation_mode
                ))
                self.connection.commit()
                cursor.close()
                print(f"生成圖片路徑已記錄: {generated_image}")
            except Error as e:
                print(f"插入生成圖片紀錄錯誤: {e}")
                self.connection = None  # 斷開連接以便下次重新連接

# 圖片處理工具
class ImageProcessor:
    def __init__(self, target_size=(256, 256)):
        self.target_size = target_size

    def preprocess_image_path(self, image_path):
        try:
            img = Image.open(image_path).convert('RGB')
            img = img.resize(self.target_size)
            img_array = np.array(img) / 255.0
            print(f"預處理圖片 {image_path} 成功，形狀: {img_array.shape}")
            return img_array
        except Exception as e:
            print(f"處理圖片錯誤 {image_path}: {e}")
            return None

    def preprocess_pil_image(self, image):
        try:
            img = image.convert('RGB')
            img = img.resize(self.target_size)
            img_array = np.array(img) / 255.0
            print(f"預處理 PIL 圖片成功，形狀: {img_array.shape}, 範圍: [{img_array.min()}, {img_array.max()}]")
            return img_array
        except Exception as e:
            print(f"處理圖片錯誤: {e}")
            return None

# 模型工具
class PortraitModel:
    def __init__(self, input_shape=(256, 256, 3), optimizer='Adam', learning_rate=0.0001, model_path='models/portrait_model.h5', model_version='v1.0'):
        self.input_shape = input_shape
        self.optimizer_name = optimizer
        self.learning_rate = learning_rate
        self.model_path = model_path
        self.model_version = model_version
        if os.path.exists(self.model_path):
            self.model = models.load_model(self.model_path, compile=False)
            print("已加載已訓練的模型")
            self.compile_model()
        else:
            self.model = self.build_model()

    def build_model(self):
        # 使用卷積層構建編碼器
        input_img = layers.Input(shape=self.input_shape)

        # 編碼器
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(input_img)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        encoded = layers.MaxPooling2D((2, 2), padding='same')(x)

        # 解碼器
        x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(encoded)
        x = layers.UpSampling2D((2, 2))(x)
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.UpSampling2D((2, 2))(x)
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.UpSampling2D((2, 2))(x)
        decoded = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

        model = models.Model(inputs=input_img, outputs=decoded)
        self.compile_model(model)
        print("基於卷積層的自動編碼器建立完成")
        return model

    def compile_model(self, model=None):
        if model is None:
            model = self.model
        if self.optimizer_name == 'Adam':
            optimizer = Adam(learning_rate=self.learning_rate)
        elif self.optimizer_name == 'RMSprop':
            optimizer = RMSprop(learning_rate=self.learning_rate)
        else:
            raise ValueError("不支持的優化器")
        model.compile(optimizer=optimizer, loss='mse', metrics=['accuracy'])
        print("模型已編譯")

    def train_on_batch(self, x):
        metrics = self.model.train_on_batch(x, x)
        loss = metrics[0]
        accuracy = metrics[1]
        return loss, accuracy

    def generate_image_autoencoder(self, input_image, parameters):
        # 使用自動編碼器生成相似圖片
        input_array = np.expand_dims(input_image, axis=0)
        print(f"輸入自動編碼器的圖片形狀: {input_array.shape}")
        generated = self.model.predict(input_array)[0]
        print(f"生成圖片的像素範圍: [{generated.min()}, {generated.max()}]")
        generated_image = (generated * 255).astype(np.uint8)
        print("自動編碼器生成圖片成功")
        return Image.fromarray(generated_image)

# Stable Diffusion 模型與增強功能
class TextToImageModel:
    def __init__(self, model_name="stabilityai/stable-diffusion-2-1", torch_dtype=torch.float32):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            huggingface_token = os.getenv("HUGGINGFACE_TOKEN")
            if not huggingface_token:
                raise ValueError("HUGGINGFACE_TOKEN not found in environment variables.")
            self.pipe_text_to_image = StableDiffusionPipeline.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                use_auth_token=huggingface_token
            )
            self.pipe_text_to_image = self.pipe_text_to_image.to(self.device)
            self.pipe_img_to_img = StableDiffusionImg2ImgPipeline.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                use_auth_token=huggingface_token
            )
            self.pipe_img_to_img = self.pipe_img_to_img.to(self.device)
            print("Stable Diffusion 模型載入成功")
        except Exception as e:
            print(f"載入模型時發生錯誤: {e}")
            self.pipe_text_to_image = None
            self.pipe_img_to_img = None

    def generate_image_text_to_image(self, prompt, num_images=1, guidance_scale=7.5, num_inference_steps=50):
        if self.pipe_text_to_image is None:
            raise ValueError("文字到圖片模型未正確加載")
        print(f"生成文字到圖片，提示: {prompt}, 張數: {num_images}")
        images = self.pipe_text_to_image(
            prompt,
            num_images_per_prompt=num_images,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps
        ).images
        print(f"生成了 {len(images)} 張文字到圖片")
        return images

    def generate_image_img_to_img(self, prompt, init_image, num_images=1, guidance_scale=7.5, num_inference_steps=50):
        if self.pipe_img_to_img is None:
            raise ValueError("圖像到圖像模型未正確加載")
        if not isinstance(init_image, Image.Image):
            raise ValueError(f"init_image 必須是 PIL.Image.Image，現在是 {type(init_image)}")
        print(f"生成圖像到圖像，提示: {prompt}, 張數: {num_images}")
        init_image = init_image.convert("RGB")
        init_image = init_image.resize((512, 512))
        print(f"初始化圖片形狀: {init_image.size}, 模式: {init_image.mode}")
        images = self.pipe_img_to_img(
            prompt=prompt,
            image=init_image,  # 更改參數名稱
            strength=0.8,      # 添加 strength 參數
            num_images_per_prompt=num_images,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps
        ).images
        print(f"生成了 {len(images)} 張圖像到圖像")
        return images

# 訓練執行緒
class TrainingThread(QThread):
    progress = pyqtSignal(int)
    log = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, image_paths, model, db, processor, trained_by='User', notes=''):
        super().__init__()
        self.image_paths = image_paths
        self.model = model
        self.db = db
        self.processor = processor
        self.trained_by = trained_by
        self.notes = notes
        self._is_running = True

    def stop(self):
        self._is_running = False

    def run(self):
        total = len(self.image_paths)
        save_interval = 100
        batch_counter = 0

        for idx, path in enumerate(self.image_paths):
            if not self._is_running:
                self.log.emit("訓練已被停止")
                break

            img = self.processor.preprocess_image_path(path)
            if img is not None:
                x = np.expand_dims(img, axis=0)
                loss, accuracy = self.model.train_on_batch(x)
                try:
                    self.db.log_training(
                        image_path=path,
                        model_version=self.model.model_version,
                        loss=float(loss),
                        accuracy=float(accuracy),
                        validation_loss=float(loss),
                        validation_accuracy=float(accuracy),
                        training_time_seconds=0,
                        trained_by=self.trained_by,
                        notes=self.notes
                    )
                except Exception as e:
                    self.log.emit(f"無法記錄訓練資料到資料庫: {e}")
                    print(f"無法記錄訓練資料到資料庫: {e}")
                batch_counter += 1

                if batch_counter % save_interval == 0:
                    self.model.model.save(self.model.model_path)
                    tf.keras.backend.clear_session()
                    self.log.emit(f"保存模型並清理記憶體 - 訓練進度: {batch_counter} 張")
                    print(f"保存模型並清理記憶體 - 訓練進度: {batch_counter} 張")

            progress = int(((idx + 1) / total) * 100)
            self.progress.emit(progress)

        if self._is_running:
            self.model.model.save(self.model.model_path)
            tf.keras.backend.clear_session()
            self.log.emit("訓練完成，最終模型已保存")
            print("訓練完成，最終模型已保存")

        self.finished.emit()

# 圖片預覽對話框
class ImagePreviewDialog(QDialog):
    def __init__(self, image_path):
        super().__init__()
        self.setWindowTitle("圖片預覽")
        self.setMinimumSize(400, 400)
        layout = QVBoxLayout()
        pixmap = QPixmap(image_path)
        label = QLabel()
        label.setPixmap(pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        layout.addWidget(label)
        self.setLayout(layout)

# 圖片生成執行緒
class ImageGenerationThread(QThread):
    progress = pyqtSignal(int)
    finished_signal = pyqtSignal(list)
    error_signal = pyqtSignal(str)

    def __init__(self, generation_mode, prompt, num_images, model, processor, init_image=None):
        super().__init__()
        self.generation_mode = generation_mode  # 'autoencoder', 'text_to_image', 'img_to_img'
        self.prompt = prompt
        self.num_images = num_images
        self.model = model
        self.processor = processor
        self.init_image = init_image  # PIL Image or None

    def run(self):
        try:
            if self.generation_mode == 'autoencoder':
                if self.init_image is None:
                    raise ValueError("初始化圖片為空")
                input_array = self.processor.preprocess_pil_image(self.init_image)
                if input_array is None:
                    raise ValueError("無法處理輸入圖片")
                print(f"Autoencoder 模式下生成 {self.num_images} 張圖片")
                generated_images = []
                for i in range(self.num_images):
                    print(f"生成自動編碼器圖片 {i+1}/{self.num_images}")
                    img = self.model.generate_image_autoencoder(input_array, {})
                    if not isinstance(img, Image.Image):
                        raise ValueError(f"生成的圖片類型錯誤: {type(img)}")
                    generated_images.append(img)
                self.progress.emit(100)
                self.finished_signal.emit(generated_images)
            elif self.generation_mode == 'text_to_image':
                # 確保 prompt 是字串
                if not isinstance(self.prompt, str):
                    self.prompt = str(self.prompt)
                print(f"Text-to-Image 模式下生成 {self.num_images} 張圖片，提示: {self.prompt}")
                images = self.model.generate_image_text_to_image(
                    prompt=self.prompt,
                    num_images=self.num_images,
                    guidance_scale=7.5,
                    num_inference_steps=50
                )
                if not all(isinstance(img, Image.Image) for img in images):
                    raise ValueError("部分生成的圖片格式錯誤")
                self.progress.emit(100)
                self.finished_signal.emit(images)
            elif self.generation_mode == 'img_to_img':
                if self.init_image is None:
                    raise ValueError("初始化圖片為空")
                if not isinstance(self.init_image, Image.Image):
                    raise ValueError(f"init_image 必須是 PIL.Image.Image，現在是 {type(self.init_image)}")
                print(f"Image-to-Image 模式下生成 {self.num_images} 張圖片，提示: {self.prompt}")
                images = self.model.generate_image_img_to_img(
                    prompt=self.prompt,
                    init_image=self.init_image,
                    num_images=self.num_images,
                    guidance_scale=7.5,
                    num_inference_steps=50
                )
                if not all(isinstance(img, Image.Image) for img in images):
                    raise ValueError("部分生成的圖片格式錯誤")
                self.progress.emit(100)
                self.finished_signal.emit(images)
            else:
                raise ValueError("未知的生成模式")
        except Exception as e:
            self.error_signal.emit(str(e))
            print(f"生成圖片時發生錯誤: {e}")

# 圖形使用者介面
class App(QWidget):
    def __init__(self):
        super().__init__()
        self.title = '人像照片訓練AI'
        self.initUI()
        self.db = Database()
        self.processor = ImageProcessor()
        self.portrait_model = PortraitModel()
        self.text_to_image_model = TextToImageModel()
        if not os.path.exists('models'):
            os.makedirs('models')
        if not os.path.exists('logs'):
            os.makedirs('logs')
        if not os.path.exists('generated'):
            os.makedirs('generated')
        self.log_file = os.path.join('logs', 'training.log')
        print("應用程序初始化完成")

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(100, 100, 1200, 900)

        layout = QVBoxLayout()

        # 訓練部分
        train_layout = QHBoxLayout()

        self.select_train_btn = QPushButton('選擇訓練圖片位置')
        self.select_train_btn.clicked.connect(self.select_train_directory)
        train_layout.addWidget(self.select_train_btn)

        self.train_path_label = QLabel('預設: C:\\Users\\User\\Pictures\\train\\pict')
        train_layout.addWidget(self.train_path_label)

        self.start_train_btn = QPushButton('開始訓練')
        self.start_train_btn.clicked.connect(self.start_training)
        train_layout.addWidget(self.start_train_btn)

        self.stop_train_btn = QPushButton('停止訓練')
        self.stop_train_btn.clicked.connect(self.stop_training)
        self.stop_train_btn.setEnabled(False)
        train_layout.addWidget(self.stop_train_btn)

        layout.addLayout(train_layout)

        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        layout.addWidget(self.log_text)

        # 生成部分
        generate_layout = QVBoxLayout()

        # 選擇生成模式
        mode_group = QGroupBox("選擇生成模式")
        mode_layout = QHBoxLayout()
        self.radio_autoencoder = QRadioButton("使用輸入圖片生成相似圖片")
        self.radio_text_to_image = QRadioButton("使用提示字生成圖片")
        self.radio_img_to_img = QRadioButton("使用輸入圖片和提示字生成圖片")
        self.radio_autoencoder.setChecked(True)
        mode_layout.addWidget(self.radio_autoencoder)
        mode_layout.addWidget(self.radio_text_to_image)
        mode_layout.addWidget(self.radio_img_to_img)
        mode_group.setLayout(mode_layout)
        generate_layout.addWidget(mode_group)

        # 上傳部分
        upload_layout = QHBoxLayout()
        self.upload_btn = QPushButton('上傳照片')
        self.upload_btn.clicked.connect(self.upload_image)
        upload_layout.addWidget(self.upload_btn)

        self.upload_preview_label = QLabel()
        self.upload_preview_label.setFixedSize(150, 150)
        self.upload_preview_label.setStyleSheet("border: 1px solid black;")
        upload_layout.addWidget(self.upload_preview_label)

        generate_layout.addLayout(upload_layout)

        # 自由文本輸入
        text_prompt_layout = QVBoxLayout()
        text_prompt_label = QLabel('輸入生成條件（最多300字）:')
        self.text_prompt_input = QTextEdit()
        self.text_prompt_input.setPlaceholderText('請輸入您的描述...')
        self.text_prompt_input.setMaximumHeight(100)
        text_prompt_layout.addWidget(text_prompt_label)
        text_prompt_layout.addWidget(self.text_prompt_input)
        generate_layout.addLayout(text_prompt_layout)

        # 生成張數
        qty_layout = QHBoxLayout()
        qty_layout.addWidget(QLabel('產生張數:'))
        self.qty_spin = QSpinBox()
        self.qty_spin.setMinimum(1)
        self.qty_spin.setMaximum(10)
        self.qty_spin.setValue(1)
        qty_layout.addWidget(self.qty_spin)
        generate_layout.addLayout(qty_layout)

        # 生成進度條
        self.generate_progress_bar = QProgressBar()
        self.generate_progress_bar.setValue(0)
        generate_layout.addWidget(self.generate_progress_bar)

        self.generate_btn = QPushButton('生成新照片')
        self.generate_btn.clicked.connect(self.generate_images)
        generate_layout.addWidget(self.generate_btn)

        # 顯示生成的圖片
        self.generated_list = QListWidget()
        self.generated_list.setViewMode(QListWidget.IconMode)
        self.generated_list.setIconSize(QSize(100, 100))
        self.generated_list.setResizeMode(QListWidget.Adjust)
        self.generated_list.itemDoubleClicked.connect(self.preview_image)
        generate_layout.addWidget(self.generated_list)

        # 保存選項
        save_layout = QHBoxLayout()
        self.save_checkbox = QCheckBox('選擇要保存的圖片')
        save_layout.addWidget(self.save_checkbox)
        self.save_btn = QPushButton('保存選擇的圖片')
        self.save_btn.clicked.connect(self.save_selected_images)
        save_layout.addWidget(self.save_btn)
        generate_layout.addLayout(save_layout)

        layout.addLayout(generate_layout)

        self.setLayout(layout)

    def select_train_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "選擇訓練圖片資料夾", "C:\\Users\\User\\Pictures\\train\\pict")
        if directory:
            self.train_path_label.setText(directory)
            self.log(f"選擇訓練資料夾: {directory}")
            print(f"選擇訓練資料夾: {directory}")

    def start_training(self):
        train_dir = self.train_path_label.text()
        if not os.path.isdir(train_dir):
            QMessageBox.warning(self, "錯誤", "請選擇有效的訓練資料夾")
            self.log("選擇的訓練資料夾無效")
            print("選擇的訓練資料夾無效")
            return
        image_extensions = ['.png', '.jpg', '.jpeg', '.bmp']
        image_paths = []
        for root, dirs, files in os.walk(train_dir):
            for file in files:
                if os.path.splitext(file)[1].lower() in image_extensions:
                    image_paths.append(os.path.join(root, file))
        if not image_paths:
            QMessageBox.warning(self, "錯誤", "訓練資料夾中沒有有效的圖片")
            self.log("訓練資料夾中沒有有效的圖片")
            print("訓練資料夾中沒有有效的圖片")
            return
        trained_by = os.getenv('TRAINED_BY', 'User')
        notes, ok = QInputDialog.getText(self, '訓練備註', '輸入訓練備註（可選）:')
        if not ok:
            notes = ''
        self.thread = TrainingThread(image_paths, self.portrait_model, self.db, self.processor, trained_by, notes)
        self.thread.progress.connect(self.update_progress)
        self.thread.log.connect(self.training_log)
        self.thread.finished.connect(self.training_complete)
        self.thread.start()
        self.start_train_btn.setEnabled(False)
        self.stop_train_btn.setEnabled(True)
        self.log("開始訓練...")
        print("開始訓練...")

    def stop_training(self):
        if hasattr(self, 'thread') and self.thread.isRunning():
            self.thread.stop()
            self.log("正在停止訓練...")
            print("正在停止訓練...")

    def update_progress(self, value):
        self.progress_bar.setValue(value)
        print(f"訓練進度: {value}%")

    def training_log(self, message):
        self.log(message)

    def training_complete(self):
        self.start_train_btn.setEnabled(True)
        self.stop_train_btn.setEnabled(False)
        self.log("訓練完成")
        QMessageBox.information(self, "完成", "訓練完成")
        print("訓練完成")

    def log(self, message):
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_entry = f"[{timestamp}] {message}"
        self.log_text.append(log_entry)
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_entry + '\n')

    def upload_image(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "選擇要上傳的照片", "", "Images (*.png *.jpg *.jpeg *.bmp)", options=options)
        if file_path:
            try:
                img = Image.open(file_path)
                img.verify()  # 驗證圖片是否損壞
                img = Image.open(file_path)  # 重新打開圖片
                img = img.convert("RGB")
                img = img.resize((512, 512))
                self.input_image_path = file_path
                pixmap = QPixmap(file_path)
                self.upload_preview_label.setPixmap(pixmap.scaled(self.upload_preview_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
                QMessageBox.information(self, "上傳成功", f"已上傳: {file_path}")
                self.log(f"上傳照片: {file_path}")
                print(f"上傳照片: {file_path}")
            except Exception as e:
                QMessageBox.warning(self, "錯誤", f"無法打開或處理上傳的圖片: {e}")
                self.log(f"無法打開或處理上傳的圖片: {e}")
                print(f"無法打開或處理上傳的圖片: {e}")

    def generate_images(self):
        generation_mode = None
        if self.radio_autoencoder.isChecked():
            generation_mode = 'autoencoder'
        elif self.radio_text_to_image.isChecked():
            generation_mode = 'text_to_image'
        elif self.radio_img_to_img.isChecked():
            generation_mode = 'img_to_img'
        else:
            QMessageBox.warning(self, "錯誤", "請選擇生成模式")
            self.log("未選擇生成模式")
            print("未選擇生成模式")
            return

        prompt = self.text_prompt_input.toPlainText().strip()
        if generation_mode in ['text_to_image', 'img_to_img']:
            if not prompt:
                QMessageBox.warning(self, "錯誤", "請輸入生成條件")
                self.log("未輸入生成條件")
                print("未輸入生成條件")
                return
            if len(prompt) > 300:
                QMessageBox.warning(self, "錯誤", "生成條件不能超過300字")
                self.log("生成條件超過300字")
                print("生成條件超過300字")
                return

        qty = self.qty_spin.value()

        init_image = None
        if generation_mode in ['autoencoder', 'img_to_img']:
            if not hasattr(self, 'input_image_path'):
                QMessageBox.warning(self, "錯誤", "請先上傳一張照片")
                self.log("未上傳照片就嘗試生成圖片")
                print("未上傳照片就嘗試生成圖片")
                return
            try:
                init_image = Image.open(self.input_image_path)
                if not isinstance(init_image, Image.Image):
                    raise ValueError("上傳的文件不是有效的圖片格式")
                self.log(f"使用上傳的圖片作為生成基礎: {self.input_image_path}")
                print(f"使用上傳的圖片作為生成基礎: {self.input_image_path}")
            except Exception as e:
                QMessageBox.warning(self, "錯誤", f"無法打開上傳的圖片: {e}")
                self.log(f"無法打開上傳的圖片: {e}")
                print(f"無法打開上傳的圖片: {e}")
                return

        # 禁用生成按鈕以防止多次點擊
        self.generate_btn.setEnabled(False)
        self.generate_progress_bar.setValue(0)
        self.log("開始生成圖片...")
        print("開始生成圖片...")

        # 根據生成模式選擇相應的模型
        if generation_mode == 'autoencoder':
            model = self.portrait_model
        else:
            model = self.text_to_image_model

        # 開始在獨立執行緒中生成圖片
        self.gen_thread = ImageGenerationThread(generation_mode, prompt, qty, model, self.processor, init_image)
        self.gen_thread.progress.connect(self.update_generate_progress)
        self.gen_thread.finished_signal.connect(lambda imgs, mode=generation_mode: self.handle_generated_images(imgs, mode))
        self.gen_thread.error_signal.connect(self.handle_generation_error)
        self.gen_thread.start()

    def update_generate_progress(self, value):
        self.generate_progress_bar.setValue(value)
        print(f"生成進度: {value}%")

    def handle_generated_images(self, images, generation_mode):
        try:
            for img in images:
                if not isinstance(img, Image.Image):
                    raise ValueError(f"生成的圖片不是 PIL.Image.Image 類型，實際類型為 {type(img)}")
                timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')
                save_path = os.path.join('generated', f'generated_{timestamp}.png')
                img.save(save_path)
                item = QListWidgetItem(QIcon(save_path), os.path.basename(save_path))
                item.setData(Qt.UserRole, save_path)
                self.generated_list.addItem(item)
                parameters = {
                    "guidance_scale": 7.5,
                    "num_inference_steps": 50
                }
                self.db.log_generated_image(
                    self.input_image_path if hasattr(self, 'input_image_path') else None,
                    save_path,
                    parameters,
                    self.text_prompt_input.toPlainText().strip(),
                    generation_mode
                )
                self.log(f"生成並保存圖片: {save_path}")
                print(f"生成並保存圖片: {save_path}")
            self.log(f"生成了 {len(images)} 張圖片")
            QMessageBox.information(self, "完成", f"生成了 {len(images)} 張圖片")
            print(f"生成了 {len(images)} 張圖片")
        except Exception as e:
            QMessageBox.critical(self, "錯誤", f"處理生成圖片時發生錯誤: {e}")
            self.log(f"處理生成圖片時發生錯誤: {e}")
            print(f"處理生成圖片時發生錯誤: {e}")
        finally:
            self.generate_btn.setEnabled(True)
            self.generate_progress_bar.setValue(100)
            print("生成圖片結束")

    def handle_generation_error(self, error_message):
        QMessageBox.critical(self, "錯誤", f"生成圖片時發生錯誤: {error_message}")
        self.log(f"生成圖片時發生錯誤: {error_message}")
        print(f"生成圖片時發生錯誤: {error_message}")
        self.generate_btn.setEnabled(True)
        self.generate_progress_bar.setValue(0)

    def preview_image(self, item):
        image_path = item.data(Qt.UserRole)
        if os.path.exists(image_path):
            dialog = ImagePreviewDialog(image_path)
            dialog.exec_()
            self.log(f"預覽圖片: {image_path}")
            print(f"預覽圖片: {image_path}")
        else:
            QMessageBox.warning(self, "錯誤", "圖片路徑不存在")
            self.log("預覽失敗，圖片路徑不存在")
            print("預覽失敗，圖片路徑不存在")

    def save_selected_images(self):
        selected_items = self.generated_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "錯誤", "請選擇要保存的圖片")
            self.log("未選擇任何圖片進行保存")
            print("未選擇任何圖片進行保存")
            return
        save_dir = QFileDialog.getExistingDirectory(self, "選擇保存位置", "C:\\Users\\User\\Pictures\\PhotoAlbums")
        if not save_dir:
            return
        for item in selected_items:
            src = item.data(Qt.UserRole)
            if os.path.exists(src):
                filename = os.path.basename(src)
                dst = os.path.join(save_dir, filename)
                try:
                    Image.open(src).save(dst)
                    self.log(f"保存圖片: {dst}")
                    print(f"保存圖片: {dst}")
                except Exception as e:
                    print(f"保存圖片錯誤 {src}: {e}")
                    self.log(f"保存圖片錯誤 {src}: {e}")
        QMessageBox.information(self, "完成", "選定的圖片已保存")
        self.log("選定的圖片已保存")
        print("選定的圖片已保存")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    ex.show()
    sys.exit(app.exec_())
