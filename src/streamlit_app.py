import os
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.callbacks import ModelCheckpoint

# Set paths
train_dir = '../data/Training'
test_dir = '../data/Testing'
model_save_path = '../model/brain_tumor_model.h5'

# Load model if available
if os.path.exists(model_save_path):
    model = load_model(model_save_path)
else:
    model = None

# Streamlit app title
st.title("Brain Tumor Detection Model")

# Sidebar parameters
st.sidebar.header("Model Parameters")
img_size = (64, 64)
batch_size = st.sidebar.slider("Batch Size", 16, 64, 32)
epochs = st.sidebar.slider("Epochs", 5, 50, 20)

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
        Input(shape=input_shape),  # Add this as the first layer
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

def plot_training_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(history.history['accuracy'], label='Train Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Test Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    
    ax2.plot(history.history['loss'], label='Train Loss')
    ax2.plot(history.history['val_loss'], label='Test Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    
    st.pyplot(fig)

# Load data and build model if necessary
if model is None:
    train_generator, test_generator = load_and_preprocess_data(train_dir, test_dir, img_size, batch_size)
    model = build_model(input_shape=(64, 64, 3), num_classes=4)

    if st.button("Train Model"):
        st.write("Training in progress...")
        history = compile_and_train_model(model, train_generator, test_generator, epochs=epochs)
        st.success("Training completed!")
        plot_training_history(history)

# Upload and predict
st.header("Brain Tumor Prediction")
uploaded_file = st.file_uploader("Upload an MRI image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = load_img(uploaded_file, target_size=img_size)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    st.image(img, caption="Uploaded MRI Image", use_column_width=True)
    
    if model:
        prediction = model.predict(img_array)
        class_idx = np.argmax(prediction, axis=1)[0]

        # Add custom class names if available
        class_names = ["No Tumor", "Type 1 Tumor", "Type 2 Tumor", "Type 3 Tumor"]
        predicted_class = class_names[class_idx]

        st.write(f"Prediction: **{predicted_class}**")
    else:
        st.write("Model not trained or loaded. Please train the model first.")