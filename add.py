import os
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input, concatenate, Dense
from tensorflow.keras.optimizers import Adam
from transformers import AutoImageProcessor, AutoModelForImageClassification
import numpy as np
import torch


class IntegrateHuggingFaceModel:
    def __init__(self, existing_model_path='models/portrait_model.h5', output_model_path='models/integrated_model.h5'):
        self.existing_model_path = existing_model_path
        self.output_model_path = output_model_path
        self.integrated_model = None
        self.existing_model = None

        # Load the Hugging Face model and processor
        self.processor = AutoImageProcessor.from_pretrained("Falconsai/nsfw_image_detection")
        self.external_model = AutoModelForImageClassification.from_pretrained("Falconsai/nsfw_image_detection")
        print("成功載入 Hugging Face 模型")

    def load_existing_model(self):
        if not os.path.exists(self.existing_model_path):
            raise FileNotFoundError(f"現有模型不存在: {self.existing_model_path}")
        self.existing_model = load_model(self.existing_model_path, compile=False)
        print(f"成功載入現有模型: {self.existing_model_path}")

    def extract_features(self, image_batch):
        """
        Passes an image batch through the Hugging Face model and extracts features.
        """
        # Preprocess the input batch
        inputs = self.processor(images=image_batch, return_tensors="pt")

        # Forward pass through Hugging Face model
        with torch.no_grad():
            outputs = self.external_model(**inputs)

        # Convert logits to NumPy array
        return outputs.logits.detach().numpy()

        def integrate_models(self):
            if self.existing_model is None:
                raise ValueError("請先載入現有模型")

            # Define the input shape
            input_shape = self.existing_model.input_shape[1:]  # Remove batch dimension
            print(f"現有模型輸入形狀: {input_shape}")

            # Create a new input layer
            input_layer = Input(shape=input_shape)

            # Process the input through the existing model
            existing_output = self.existing_model(input_layer)

            # Flatten the existing model output
            existing_output_flattened = tf.keras.layers.Flatten()(existing_output)

            # Determine the output shape of the Hugging Face model
            sample_input = np.random.rand(1, *input_shape)  # Generate a sample input
            processor_output = self.processor(images=sample_input, return_tensors="pt")
            hugging_face_output = self.external_model(**processor_output).logits.detach().numpy()
            output_shape = hugging_face_output.shape[1:]  # Remove batch dimension
            print(f"Hugging Face 模型輸出形狀: {output_shape}")

            # Hugging Face model feature extraction
            hugging_face_features = tf.keras.layers.Lambda(
                lambda x: tf.convert_to_tensor(self.extract_features(x), dtype=tf.float32),
                output_shape=output_shape
            )(input_layer)

            # Flatten Hugging Face model output
            hugging_face_features_flattened = tf.keras.layers.Flatten()(hugging_face_features)

            # Merge the outputs
            merged_output = concatenate([existing_output_flattened, hugging_face_features_flattened])

            # Add a dense layer for final output
            final_output = Dense(256, activation='relu')(merged_output)  # Optional: Adjust units
            final_output = Dense(10, activation='softmax')(final_output)  # Adjust output size as needed

            # Create the final integrated model
            self.integrated_model = Model(inputs=input_layer, outputs=final_output)
            print("成功整合 Hugging Face 模型與現有模型")


    def compile_and_save(self, learning_rate=0.0001):
        if self.integrated_model is None:
            raise ValueError("請先整合模型")

        # Compile the integrated model
        self.integrated_model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse', metrics=['accuracy'])
        print("整合模型編譯完成")

        # Save the integrated model
        self.integrated_model.save(self.output_model_path)
        print(f"整合模型已保存至: {self.output_model_path}")


if __name__ == '__main__':
    # Define model paths
    existing_model_path = 'models/portrait_model.h5'
    output_model_path = 'models/integrated_model.h5'

    # Create an instance of the integration class
    integrator = IntegrateHuggingFaceModel(existing_model_path, output_model_path)

    # Ensure the following methods exist and are called in sequence
    integrator.load_existing_model()  # Load the existing Keras model
    integrator.integrate_models()    # Integrate with Hugging Face model
    integrator.compile_and_save()    # Compile and save the integrated model