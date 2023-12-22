import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

train_path = 'fruits-360/Training'
test_path = 'fruits-360/Test'

# data preprocessing
train_datagen = ImageDataGenerator(rescale=1./255, 
                                   shear_range=0.2, 
                                   zoom_range=0.2, 
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_path, 
                                                    target_size=(100, 100), 
                                                    batch_size=32, 
                                                    class_mode='categorical')

test_generator = test_datagen.flow_from_directory(test_path, 
                                                  target_size=(100, 100), 
                                                  batch_size=32, 
                                                  class_mode='categorical')

# ResNet50, pre-trained on ImageNet
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(100, 100, 3))

# building neural network
x = base_model.output

# global average pooling to reduce number of features
x = GlobalAveragePooling2D()(x)

# less neurons to reduce computational complexity
x = Dense(1024, activation='relu')(x)

# for faster convergence
x = BatchNormalization()(x)

# softmax activation function in last layer
predictions = Dense(train_generator.num_classes, activation='softmax')(x)

# target model
model = Model(inputs=base_model.input, outputs=predictions)

# train top layers first (randomly initialized!!)
for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# training
history = model.fit(train_generator, 
                    steps_per_epoch=train_generator.samples//train_generator.batch_size, 
                    epochs=10, 
                    validation_data=test_generator, 
                    validation_steps=test_generator.samples//test_generator.batch_size)

# plot history
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')

# evaluation
test_loss, test_acc = model.evaluate(test_generator, steps=test_generator.samples//test_generator.batch_size)
print(f"Test Accuracy: {test_acc}")

# save model
model.save('fruits.h5')
