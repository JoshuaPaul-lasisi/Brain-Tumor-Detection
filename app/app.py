from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io

app = Flask(__name__)
model = load_model('model/brain_tumor_model.h5')

def preprocess_image(image):
    image = image.resize((64, 64))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    image = Image.open(file.stream)
    image = preprocess_image(image)

    prediction = model.predict(image)
    class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']
    result = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)

    return jsonify({'prediction': result, 'confidence': confidence})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)