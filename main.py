import argparse

parser = argparse.ArgumentParser(
        description='This application is used to use a model for fruit classification in real time via the camera. Exit by pressing q.')

parser.add_argument('-m', '--model', type=str, default='fruits.h5',
                    help='Choose what model should be used. Specify the path of the model here.')
parser.add_argument('-tr', '--train_path', type=str, default='fruits-360/Training',
                    help='Specify the path of the training dataset here. Is required to identify the names of the classes.')
parser.add_argument('-c', '--camera', type=int, default=0,
                    help='Choose the index of the camera here. By default, the first camera found is chosen.')

args = parser.parse_args()


import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image

model = load_model(args.model)

temp_datagen = ImageDataGenerator(rescale=1./255)
temp_generator = temp_datagen.flow_from_directory(
    args.train_path,
    target_size=(100, 100),
    batch_size=32,
    class_mode='categorical',
    shuffle=False)

class_names = list(temp_generator.class_indices.keys())

cap = cv2.VideoCapture(args.camera)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # preprocess image
    # resize image
    resized_frame = cv2.resize(frame, (100, 100))
    # convert image color to array
    img_array = image.img_to_array(resized_frame)
    # normalize the image
    img_array = img_array / 255.0
    # expand dimensions to match batch size
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)

    sorted_indices = np.argsort(predictions[0])[::-1]

    top_n = 10
    for i in range(top_n):
        idx = sorted_indices[i]
        probability = predictions[0][idx]
        cv2.putText(frame, 
                    f'{class_names[idx]} ({probability:.2f}%)', 
                    (10, 30 + i*35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1, 
                    (0, 255, 0), 
                    2)

    cv2.imshow('Frame', frame)

    # press 'q' to exit loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
