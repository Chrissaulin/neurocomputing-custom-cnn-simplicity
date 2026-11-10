import os
import sys

os.environ["KERAS_HOME"] = os.path.abspath(".keras")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import argparse
import numpy as np
from PIL import Image
import tensorflow as tf

CLASS_NAMES = [
    "people",
    "beach",
    "building",
    "bus",
    "dinosaur",
    "elephant",
    "flower",
    "horse",
    "mountain",
    "food"
]

def load_and_preprocess_image(image_path, target_size=(128, 128)):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"File tidak ditemukan: {image_path}")
    image = Image.open(image_path).convert("RGB")
    image = image.resize(target_size)
    image_array = np.array(image, dtype=np.float32) / 255.0
    return np.expand_dims(image_array, axis=0)

def main():
    parser = argparse.ArgumentParser(description="Inference CLI for SIMPLIcity Custom CNN")
    parser.add_argument("--image", type=str, required=True, help="Path ke file citra pengujian")
    parser.add_argument("--model", type=str, default="models/best_custom_cnn_simplicity.keras", help="Path ke file model")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Error: Model file tidak ditemukan di {args.model}")
        sys.exit(1)

    model = tf.keras.models.load_model(args.model, compile=False)
    processed_input = load_and_preprocess_image(args.image)

    predictions = model.predict(processed_input, verbose=0)[0]
    predicted_index = int(np.argmax(predictions))
    predicted_label = CLASS_NAMES[predicted_index]
    confidence_score = float(predictions[predicted_index]) * 100.0

    print("=" * 45)
    print("HASIL INFERENSI CITRA - SIMPLIcity CNN")
    print("=" * 45)
    print(f"File Citra       : {args.image}")
    print(f"Prediksi Kelas   : {predicted_label.upper()}")
    print(f"Tingkat Keyakinan: {confidence_score:.2f}%")
    print("-" * 45)
    print("Distribusi Probabilitas Kelas:")
    for idx, (cls_name, prob) in enumerate(zip(CLASS_NAMES, predictions)):
        bar = "#" * int(prob * 20)
        print(f"{cls_name:<10} : {prob * 100:6.2f}% | {bar}")
    print("=" * 45)

if __name__ == "__main__":
    main()
