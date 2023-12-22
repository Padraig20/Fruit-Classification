import argparse

parser = argparse.ArgumentParser(
        description='This application is used to use a model for fruit classification for a single picture.')

parser.add_argument('-m', '--model', type=str, default='fruits.h5',
                    help='Choose what model should be used. Specify the path of the model here.')
parser.add_argument('-tr', '--train_path', type=str, default='fruits-360/Training',
                    help='Specify the path of the training dataset here. Is required to identify the names of the classes.')
parser.add_argument('image', metavar='image', type=str, nargs=1,
                    help='The path to the image you want to classify.')

args = parser.parse_args()

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import numpy as np

model = load_model(args.model)

temp_datagen = ImageDataGenerator(rescale=1./255)
temp_generator = temp_datagen.flow_from_directory(
    args.train_path,
    target_size=(100, 100),
    batch_size=32,
    class_mode='categorical',
    shuffle=False)

class_names = list(temp_generator.class_indices.keys())

# load image, predict
img_path = args.image[0]
img = image.load_img(img_path, target_size=(100, 100))
img_array = image.img_to_array(img)
img_array = img_array / 255.0
img_array = np.expand_dims(img_array, axis=0)

predictions = model.predict(img_array)

sorted_indices = np.argsort(predictions[0])[::-1]

# display top 10 predictions
top_n = 10
for i in range(top_n):
    idx = sorted_indices[i]
    probability = predictions[0][idx]
    print(f"{class_names[idx]}: {probability * 100:.2f}%")
