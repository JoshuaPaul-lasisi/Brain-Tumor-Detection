# Import necessary modules
import os
from data_preprocessing import load_and_preprocess_data
from model_building import build_model, compile_and_train_model

# Define directories for train and test data
train_dir = '../data/Training'
test_dir = '../data/Testing'
processed_data_dir = '../data/processed'
model_save_path = '../model/brain_tumor_model.h5'

# Ensure processed data and model directories exist
os.makedirs(processed_data_dir, exist_ok=True)
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

# Load and preprocess the data, saving it to processed_data_dir
train_generator, test_generator = load_and_preprocess_data(
    train_dir=train_dir,
    test_dir=test_dir,
    save_processed_data=True,  # Modify load_and_preprocess_data function to save preprocessed data
    processed_data_dir=processed_data_dir
)

# Build and compile the model
model = build_model(input_shape=(64, 64, 3), num_classes=4)

# Train the model and save it to model_save_path
history = compile_and_train_model(
    model,
    train_generator,
    test_generator,
    epochs=20,
    save_path=model_save_path
)