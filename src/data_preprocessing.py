import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pickle

def load_and_preprocess_data(train_dir, test_dir, img_size=(64, 64), batch_size=32, save_processed_data=False, processed_data_dir=None):
    # Data augmentation for the training set
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

    # Only rescaling for the test set
    test_datagen = ImageDataGenerator(rescale=1.0/255)

    # Create the data generators
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

    # Save the generators if needed
    if save_processed_data and processed_data_dir:
        with open(os.path.join(processed_data_dir, 'train_data.pkl'), 'wb') as train_file:
            pickle.dump(train_generator, train_file)
        with open(os.path.join(processed_data_dir, 'test_data.pkl'), 'wb') as test_file:
            pickle.dump(test_generator, test_file)

    return train_generator, test_generator