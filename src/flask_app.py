import os
import numpy as np
from flask import Flask, request, render_template, jsonify
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.callbacks import ModelCheckpoint

# Initialize Flask app
app = Flask(__name__)

# Set paths
train_dir = '../data/Training'
test_dir = '../data/Testing'
model_save_path = '../model/brain_tumor_model.h5'

# Load model if available
if os.path.exists(model_save_path):
    model = load_model(model_save_path)
else:
    model = None

# Model and data processing functions
def load_and_preprocess_data(train_dir, test_dir, img_size=(64, 64), batch_size=32):
    train_datagen = ImageDataGenerator(
        rescale=1.0/255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    test_datagen = ImageDataGenerator(rescale=1.0/255)
    
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical'
    )
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical'
    )
    return train_generator, test_generator

def build_model(input_shape=(64, 64, 3), num_classes=4):
    model = Sequential([
        Input(shape=input_shape),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    return model

def compile_and_train_model(model, train_generator, test_generator, epochs=20, save_path=model_save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    checkpoint = ModelCheckpoint(save_path, monitor='val_accuracy', save_best_only=True, verbose=1)

    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=test_generator,
        callbacks=[checkpoint]
    )
    return history

# Flask Routes
@app.route('/')
def index():
    return render_template('index.html')  # HTML form for uploading an image and training the model

@app.route('/train', methods=['POST'])
def train_model():
    global model
    if model is None:
        train_generator, test_generator = load_and_preprocess_data(train_dir, test_dir, img_size=(64, 64), batch_size=32)
        model = build_model(input_shape=(64, 64, 3), num_classes=4)
        
        # Trigger training
        history = compile_and_train_model(model, train_generator, test_generator, epochs=20)
        return "Training completed successfully!"
    return "Model already trained."

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    img = load_img(file, target_size=(64, 64))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    if model:
        prediction = model.predict(img_array)
        class_idx = np.argmax(prediction, axis=1)[0]

        class_names = ["No Tumor", "Type 1 Tumor", "Type 2 Tumor", "Type 3 Tumor"]
        predicted_class = class_names[class_idx]

        return jsonify({'Prediction': predicted_class})
    else:
        return jsonify({'error': 'Model not trained or loaded'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)