import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_and_preprocess_data(train_dir, test_dir, output_dir='../data/processed', img_size=(64, 64), batch_size=32):
    os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists
    train_output_dir = os.path.join(output_dir, 'train')
    test_output_dir = os.path.join(output_dir, 'test')
    os.makedirs(train_output_dir, exist_ok=True)
    os.makedirs(test_output_dir, exist_ok=True)
    
    # Data augmentation for the training set
    train_datagen = ImageDataGenerator(
        rescale=1.0/255,        # Normalize pixel values
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Only rescaling for the test set
    test_datagen = ImageDataGenerator(rescale=1.0/255)

    # Create the data generators
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    # Save processed training data
    for i, (images, labels) in enumerate(train_generator):
        if i * batch_size >= train_generator.samples:
            break  # Stop after one full pass through the data
        np.save(os.path.join(train_output_dir, f'train_images_batch_{i}.npy'), images)
        np.save(os.path.join(train_output_dir, f'train_labels_batch_{i}.npy'), labels)

    # Save processed test data
    for i, (images, labels) in enumerate(test_generator):
        if i * batch_size >= test_generator.samples:
            break  # Stop after one full pass through the data
        np.save(os.path.join(test_output_dir, f'test_images_batch_{i}.npy'), images)
        np.save(os.path.join(test_output_dir, f'test_labels_batch_{i}.npy'), labels)

    return train_generator, test_generator