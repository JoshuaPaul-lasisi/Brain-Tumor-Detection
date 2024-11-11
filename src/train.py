import os
from src.data_preprocessing import load_and_preprocess_data
from src.model_building import build_model, compile_and_train_model

# Paths to the directories containing training and testing data
train_dir = 'data/Training'
test_dir = 'data/Testing'

# Load and preprocess the data
train_generator, test_generator = load_and_preprocess_data(train_dir, test_dir)

# Build the model
model = build_model(input_shape=(64, 64, 3), num_classes=4)

# Compile and train the model, and save the best model during training
history = compile_and_train_model(model, train_generator, test_generator, epochs=20, save_path='model/brain_tumor_model.h5')