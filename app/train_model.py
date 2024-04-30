import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import json

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

# Custom Data Generator to handle specific labeling format
class CustomDataGenerator(ImageDataGenerator):
    def flow_from_directory(self, directory, *args, **kwargs):
        generator = super().flow_from_directory(directory, *args, **kwargs)
        generator.class_indices = self._get_class_indices(generator.filenames)
        generator.classes = self._get_classes(generator.filenames, generator.class_indices)
        return generator

    def _get_class_indices(self, filenames):
        classes = [self._parse_label(os.path.dirname(filename)) for filename in filenames]
        unique_classes = sorted(set(classes))
        return {cls: idx for idx, cls in enumerate(unique_classes)}

    def _get_classes(self, filenames, class_indices):
        return np.array([class_indices[self._parse_label(os.path.dirname(filename))] for filename in filenames])

    def _parse_label(self, label):
        parts = label.split('_')
        return ' '.join(parts[-2:])  

def create_model(num_classes):
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model():
    num_classes = 4147  
    model = create_model(num_classes)

    train_datagen = CustomDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2
    )
    validation_datagen = CustomDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        'D:/PlantAI/data/train_mini',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical'
    )

    validation_generator = validation_datagen.flow_from_directory(
        'D:/PlantAI/data/val',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical'
    )

    model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=1,
        validation_data=validation_generator,
        validation_steps=len(validation_generator)
    )
    model.save('D:/PlantAI/models/my_model.h5')

    # Save class indices to a JSON file
    class_indices = train_generator.class_indices
    with open('D:/PlantAI/models/class_indices.json', 'w') as json_file:
        json.dump(class_indices, json_file)

if __name__ == '__main__':
    train_model()