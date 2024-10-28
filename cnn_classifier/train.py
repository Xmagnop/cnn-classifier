import os
from PIL import Image
import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras import layers, models
import matplotlib.pyplot as plt

from params import epochs, optimizer, loss

le = LabelEncoder()


def load_imgs_arr_labels(folder: str):
  imgs_arr = []
  for filename in os.listdir(folder):
    img_path = os.path.join(folder, filename)
    img = Image.open(img_path)
    imgs_arr.append(np.array(img))

  label = folder.split('/')[-1]
  return np.array(imgs_arr), np.array([label] * len(imgs_arr))


def plot_training_history(history):
  # Plot da acurácia
  plt.figure(figsize=(14, 5))

  # Plot da Acurácia
  plt.subplot(1, 2, 1)
  plt.plot(history.history['accuracy'], label='Acurácia de Treinamento')
  plt.plot(history.history['val_accuracy'], label='Acurácia de Validação')
  plt.xlabel('Épocas')
  plt.ylabel('Acurácia')
  plt.legend(loc='lower right')
  plt.title('Acurácia de Treinamento e Validação')

  # Plot da Perda
  plt.subplot(1, 2, 2)
  plt.plot(history.history['loss'], label='Perda de Treinamento')
  plt.plot(history.history['val_loss'], label='Perda de Validação')
  plt.xlabel('Épocas')
  plt.ylabel('Perda')
  plt.legend(loc='upper right')
  plt.title('Perda de Treinamento e Validação')

  plt.show()


def run():
  print('Reading images...')
  imgs_arr: NDArray = None
  imgs_labels: NDArray = None
  for path in os.listdir('tmp/processed_imgs'):
    arr, labels = load_imgs_arr_labels(f'tmp/processed_imgs/{path}')
    if imgs_arr is None:
      imgs_arr, imgs_labels = arr, labels
    else:
      imgs_arr = np.concatenate([imgs_arr, arr])
      imgs_labels = np.concatenate([imgs_labels, labels])

  print('Encoding labels...')
  encoded_imgs_labels = le.fit_transform(imgs_labels)

  train_imgs_arr, test_imgs_arr, train_labels, test_labels = train_test_split(
      imgs_arr, encoded_imgs_labels, test_size=0.2)

  print('Creating model...')
  num_classes = len(os.listdir('imgs'))
  input_shape = train_imgs_arr.shape[1:]
  model = models.Sequential()

  model.add(layers.Rescaling(1. / 255, input_shape=input_shape))
  data_augmentation = models.Sequential(
      [
          layers.RandomFlip("horizontal"),
          layers.RandomRotation(0.1),
          layers.RandomZoom(0.1),
      ]
  )
  model.add(data_augmentation)

  # Camadas de convolução
  model.add(layers.Conv2D(16, 3, activation='relu'))
  model.add(layers.MaxPooling2D())
  model.add(layers.Conv2D(32, 3, activation='relu'))
  model.add(layers.MaxPooling2D())
  model.add(layers.Conv2D(64, 3, activation='relu'))
  model.add(layers.MaxPooling2D())

  model.add(layers.Dropout(0.2))
  # Linearização de array de output
  model.add(layers.Flatten())
  # Camadas finais densas
  model.add(layers.Dense(128, activation='relu'))
  model.add(layers.Dense(num_classes, activation='softmax'))

  model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

  print('Training model...')
  history = model.fit(
      train_imgs_arr,
      train_labels,
      epochs=epochs,
      validation_data=(test_imgs_arr, test_labels)
  )

  # Avaliar o modelo no conjunto de teste
  _, test_acc = model.evaluate(test_imgs_arr, test_labels)
  print('\nTest accuracy:', test_acc)

  # Exibir gráficos da performance
  plot_training_history(history)

  print('Exporting model...')
  model.save('tmp/model')
