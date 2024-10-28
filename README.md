# Sobre o Projeto

Este é projeto de um pacote Python capaz de treinar um classificador CNN(Convolutional Neural Network) de imagens.

# Obtenção de Imagens

Sites como o [Kaggle](https://www.kaggle.com/) e [images.cv](https://images.cv/) podem ser usados para obter a base de dados das imagens. Note que pode ser interessante remover ou filtrar algumas imagens obtidas destes sites por não serem ideais para certas aplicações de ML.

# Configuração de Parâmetros

É possível configurar os parâmetros de pré-processamento e treino pelo arquivo `params.py`. Segue abaixo um exemplo desse arquivo.

```python
color_mode = 'RGB'
img_size = 32

optimizer = 'adam'
loss = 'sparse_categorical_crossentropy'
epochs = 50

fine_tune_epochs = 10
fine_tune_num_layers = 8

test_model = "fine_tunel_model"
```

* `color_mode`: Modo de cor das imagens pré-processadas. Pode ser 'L' para escala de cinza e 'RGB' para a escala RGB
* `img_size`: Valor para o qual as imagens serão redimensionadas
* `optimizer`: Algoritmo usado para ajustar pesos do modelo durante treinamento. Ver mais [aqui](https://keras.io/api/optimizers/)
* `loss`: Função de perda (ou função custo) que mede o quanto as previsões do modelo estão se afastando dos valores esperados. Ver mais [aqui](https://keras.io/api/losses/)
* `epochs`: Quantidade de épocas do treinamento do zero usando Tensorflow
* `fine_tune_epochs`: Quantidade de épocas de treinamento no treinamento usando *transfer learning*
* `fine_tune_num_layers`: Número de camadas que serão descongeladas e treinadas no treinamento usando *transfer learning*
* `test_model`: Modelo que será usado na predição. Pode ser `model` para o modelo treinado do zero ou `fine_tune_model` para o modelo treinado usando *transfer learning*

As imagens de treinamento devem ser salvas na pasta `imgs`. Esta pasta deve conter subpastas que por sua vez conterão as imagens. O nome de cada subpasta irá representar o nome da classe das imagens contidas nela. Neste projeto, a pasta `imgs` contem as subpastas `ensolarado` e `chuvoso`, contendo imagens do céu nestes respectivos climas.

A pasta `predict_imgs` deve conter imagens definidas pelo usuário que serão usadas para testar o modelo pronto. Por padrão, esta pasta contém imagens de flores dos tipos `daisy`(margaridas), `rose`(rosas) e `rapeseed`(colzas).

# Executando o Código

## Instalando Depedências

`poetry install`

## Realizando Pré-processamento

`poetry run preprocess`

Este comando irá gerar a pasta `tmp` contendo as imagens pré-processadas.

## Realizando Treinamento

`poetry run train`

Este comando irá realizar o treinamento e ao fim fazer o plot das métricas obtidas durante o treinamento e validação. Após isto, irá exportar o modelo treinado para `tmp/model`.

OU

`poetry run fine_tune`

Este comando irá exportar o modelo treinado para `tmp/fine_tunel_model`.

## Classificando Imagens

`poetry run predict`

Este comando irá classificar as imagens geradas na pasta `predict_imgs`, printando os resultados a exemplo do visto abaixo.

```
chuvoso_4.jpg: chuvoso (99.98%)
ensolarado_3.jpg: chuvoso (58.72%)
chuvoso_5.jpg: chuvoso (99.98%)
ensolarado_5.jpg: ensolarado (94.27%)
ensolarado_4.jpg: ensolarado (98.01%)
ensolarado_2.jpg: ensolarado (99.68%)
chuvoso_1.jpg: chuvoso (99.88%)
ensolarado_1.jpg: ensolarado (98.55%)
chuvoso_2.jpg: chuvoso (99.88%)
chuvoso_3.jpg: chuvoso (99.91%)
```
