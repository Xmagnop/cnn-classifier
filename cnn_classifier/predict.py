import os
from PIL import Image
import numpy as np
from tensorflow import keras
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from params import img_size, color_mode, test_model


def run():
  # Carregar o modelo
  model = keras.models.load_model(f'tmp/{test_model}')

  # Obter labels e listas para armazenar as imagens e seus nomes
  labels = os.listdir('imgs')
  filenames = []
  imgs_arrs = []

  # Carregar imagens para predição
  for filename in os.listdir('predict_imgs'):
    if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
      img_path = f'predict_imgs/{filename}'
      filenames.append(filename)
      with Image.open(img_path) as img:
        img = img.convert(color_mode)
        img = img.resize((img_size, img_size))
        img_arr = np.array(img)
        imgs_arrs.append(img_arr)

  # Realizar as predições
  predictions = model.predict(np.array(imgs_arrs))

  # Converter as predições em labels e probabilidades
  predicted_labels = [np.argmax(pred) for pred in predictions]
  predicted_probs = [np.max(pred) for pred in predictions]
  # Exemplo para labels verdadeiras
  true_labels = [labels.index(filename.split('_')[0])
                 for filename in filenames]

  # Cálculo de métricas
  acc = accuracy_score(true_labels, predicted_labels)
  f1 = f1_score(true_labels, predicted_labels, average='weighted')
  conf_matrix = confusion_matrix(true_labels, predicted_labels)
  report = classification_report(
      true_labels, predicted_labels, target_names=labels)

  # Exibir as métricas no console
  print("Acurácia:", acc)
  print("F1 Score:", f1)
  print("\nRelatório de Classificação:\n", report)

  # Exibir as previsões
  for filename, pred_label, prob in zip(filenames, predicted_labels, predicted_probs):
    print(f'{filename}: {labels[pred_label]} ({prob * 100:.2f}%)')

  # Plotar matriz de confusão
  plt.figure(figsize=(10, 8))
  sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
              xticklabels=labels, yticklabels=labels)
  plt.xlabel("Predicted Label")
  plt.ylabel("True Label")
  plt.title("Matriz de Confusão")
  plt.show()

  # Gráfico de barras para visualizar as predições
  plt.figure(figsize=(12, 6))
  sns.barplot(x=filenames, y=predicted_probs, hue=[
              labels[i] for i in predicted_labels])
  plt.xticks(rotation=90)
  plt.ylabel("Probabilidade")
  plt.title("Confiança das Predições por Imagem")
  plt.legend(title="Label Previsto", bbox_to_anchor=(1, 1))
  plt.show()
