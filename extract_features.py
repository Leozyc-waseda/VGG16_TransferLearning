import os, random
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing import image
import tensorflow as tf

base_dir = '/home/ogai/Downloads/dogs-vs-cats'
train_dir = os.path.join(base_dir, 'train_new')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')
train_cats_dir = os.path.join(train_dir, 'cats')
train_dogs_dir = os.path.join(train_dir, 'dogs')

train_size, validation_size, test_size = 9000, 2300, 1300

img_width, img_height = 224, 224  # Default input size for VGG16


def show_pictures(path):
  random_img = random.choice(os.listdir(path))
  img_path = os.path.join(path, random_img)
  img = image.load_img(img_path, target_size=(img_width, img_height))
  img_tensor = image.img_to_array(img)  # Image data encoded as integers in the 0–255 range
  img_tensor /= 255.  # Normalize to [0,1] for plt.imshow application
  plt.imshow(img_tensor)
  plt.show()

# Instantiate convolutional base
from keras.applications import VGG16

conv_base = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(img_width, img_height, 3))  # 3 = number of channels in RGB pictures
# Freeze all
for layer in conv_base.layers:
    layer.trainable = False
#Fine-tuning the model
# set_trainable = False
# for layer in conv_base.layers:
#     if layer.name in ['block5_conv1', 'block4_conv1']:  set_trainable = True
#     if set_trainable:  layer.trainable = True
#     else:  layer.trainable = False
conv_base.summary()

#pass our images through it for feature extraction
# Extract features
import os, shutil
from keras.preprocessing.image import ImageDataGenerator

#data generators
# train_datagen =  ImageDataGenerator(rescale=1./255,
#                                     zoom_range=0.3,
#                                     rotation_range=50,
#                                     width_shift_range=0.2,
#                                     height_shift_range=0.2,
#                                     shear_range=0.2,   horizontal_flip=True,
#                                     fill_mode='nearest')


datagen = ImageDataGenerator(rescale=1. / 255)
batch_size = 20


# def train_extract_features(directory, sample_count):
#   features = np.zeros(shape=(sample_count, 7, 7, 512))  # Must be equal to the output of the convolutional base
#   labels = np.zeros(shape=(sample_count))
#   # Preprocess data
#   generator = train_datagen.flow_from_directory(directory,
#                                           target_size=(img_width, img_height),
#                                           shuffle=True,
#                                           batch_size=batch_size,
#                                           class_mode='binary')
#   # Pass data through convolutional base
#   i = 0
#   for inputs_batch, labels_batch in generator:
#     features_batch = conv_base.predict(inputs_batch)
#     # ここでは特徴（feature）と正解（label）のリストに
#     # 指定フォルダから読み込んだデータをバッチサイズずつ詰め込んでいます。
#     features[i * batch_size: (i + 1) * batch_size] = features_batch
#     labels[i * batch_size: (i + 1) * batch_size] = labels_batch
#     i += 1
#     if i * batch_size >= sample_count:
#       break
#   return features, labels

def extract_features(directory, sample_count):
  features = np.zeros(shape=(sample_count, 7, 7, 512))  # Must be equal to the output of the convolutional base
  labels = np.zeros(shape=(sample_count))
  # Preprocess data
  generator = datagen.flow_from_directory(directory,
                                          target_size=(img_width, img_height),
                                          batch_size=batch_size,
                                          class_mode='binary')
  # Pass data through convolutional base
  i = 0
  for inputs_batch, labels_batch in generator:
    features_batch = conv_base.predict(inputs_batch)
    # ここでは特徴（feature）と正解（label）のリストに
    # 指定フォルダから読み込んだデータをバッチサイズずつ詰め込んでいます。
    features[i * batch_size: (i + 1) * batch_size] = features_batch
    labels[i * batch_size: (i + 1) * batch_size] = labels_batch
    i += 1
    if i * batch_size >= sample_count:
      break
  return features, labels


train_features, train_labels = extract_features(train_dir, train_size)  # Agree with our small dataset size
validation_features, validation_labels = extract_features(validation_dir, validation_size)
test_features, test_labels = extract_features(test_dir, test_size)



#on top of our convolutional base,
#we will add a classifier and then our model is ready to make predictions.
from keras import models
from keras import layers
from keras import optimizers

EPOCHS = 100

model = models.Sequential()
model.add(layers.Flatten(input_shape=(7,7,512)))
model.add(layers.Dense(256, activation='relu', input_dim=(7*7*512)))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(1, activation='sigmoid'))

model.summary()
# Compile model
model.compile(optimizer=optimizers.RMSprop(lr=2e-5),
              # optimizers.Adam(),
              loss='binary_crossentropy',
              metrics=['acc'])
# Train model
#fit(x,y) x is input data, y is numpy array labels
history = model.fit(train_features, train_labels,
                    epochs=EPOCHS,
                    batch_size=batch_size,
                    validation_data=(validation_features, validation_labels),
                    verbose=1)


# Save model
model.save('/home/ogai/Downloads/dogs-vs-cats/dogs_cat_fcl.h5')

import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(EPOCHS)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


test_cats_dir = '/home/ogai/Downloads/dogs-vs-cats/test/cats'
test_dogs_dir = '/home/ogai/Downloads/dogs-vs-cats/test/dogs'

# Define function to visualize predictions
class_names = np.array(test_labels)
print(class_names)

def visualize_predictions(classifier, n_cases):
    plt.figure(figsize=(10, 9))
    for i in range(0,n_cases):
        path = random.choice([test_cats_dir, test_dogs_dir])
        # Get picture
        random_img = random.choice(os.listdir(path))
        img_path = os.path.join(path, random_img)
        img = image.load_img(img_path, target_size=(img_width, img_height))
        img_tensor = image.img_to_array(img)  # Image data encoded as integers in the 0–255 range
        img_tensor /= 255.  # Normalize to [0,1] for plt.imshow application
        # print("img_path:",img_path,"random_img:",random_img,"img_tensor",img_tensor)
        # Extract features
        features = conv_base.predict(img_tensor.reshape(1,img_width, img_height, 3))
        # Make prediction
        try:
            prediction = classifier.predict(features)
        except:
            prediction = classifier.predict(features.reshape(1, 7*7*512))

        # Show picture
        plt.subplot(5, 4, i + 1)
        plt.subplots_adjust(hspace=0.3)
        plt.axis('off')
        plt.imshow(img_tensor)
        _ = plt.suptitle("Model predictions (blue: correct, red: incorrect)")

        if 'cat.' in random_img:
            labels_img = 0
        else:
            labels_img = 1

        # Write prediction
        if prediction < 0.5:
            prediction_labels = 0
            color = "blue" if labels_img == prediction_labels else "red"
            plt.title("Cat",color = color)
            print('Cat')
        else:
            prediction_labels = 1
            color = "blue" if labels_img == prediction_labels else "red"
            plt.title("Dog",color = color)
            print('Dog')




# Visualize predictions
visualize_predictions(model, 20)