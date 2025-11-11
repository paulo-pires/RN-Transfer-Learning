# 🚀 Meu Estudo de Classificação de Gatos e Cachorros com Transfer Learning (MobileNetV2)

## 📖 Descrição do Estudo

Neste projeto, realizei um estudo prático de **Transfer Learning** para resolver um problema clássico de classificação de imagens: diferenciar gatos de cachorros. Meu objetivo era construir um modelo de alta precisão sem o custo computacional de treinar uma rede neural profunda do zero.

Para isso, utilizei o modelo pré-treinado **MobileNetV2** e o adaptei para esta tarefa.

Um dos principais desafios que enfrentei foi o tempo de treinamento. O dataset completo é grande e uma única época poderia levar horas. Para resolver isso e conseguir iterar mais rápido, decidi usar uma estratégia de **prototipagem rápida**, utilizando apenas **15%** do conjunto de dados para o treinamento inicial.

## 💡 Conceitos que Apliquei

Este projeto foi centrado em dois conceitos fundamentais de Deep Learning:

### Transfer Learning (Feature Extraction)

* **O que eu fiz:** Em vez de começar do zero, eu carreguei o MobileNetV2 já treinado na base de dados ImageNet.
* **Por que fiz isso?** Esse modelo já "sabe" identificar características visuais complexas (bordas, texturas, formas). Eu "congelei" os pesos dessas camadas e apenas treinei uma nova camada de classificação no topo, que adicionei manualmente.
* **Vantagem:** Isso reduziu drasticamente o tempo de treinamento e a necessidade de dados, me permitindo alcançar uma alta precisão rapidamente.

### Fine-Tuning (Ajuste Fino)

* **O que eu fiz:** Após o primeiro treinamento, eu "descongelei" algumas das camadas superiores do MobileNetV2.
* **Por que fiz isso?** Isso permitiu que o modelo ajustasse levemente suas características mais abstratas para se especializarem no meu problema (diferenciar gatos de cachorros, em vez de 1000 classes genéricas).
* **Como?** Eu continuei o treinamento, mas com uma **taxa de aprendizado (learning rate) muito baixa**. Isso foi crucial para não "estragar" o conhecimento valioso que o modelo já possuía.

## 💾 O Conjunto de Dados: `cats_vs_dogs`

* **Fonte:** `tensorflow_datasets` (TFDS).
* **Desafio:** O dataset original só tem um split de `train` (cerca de 23.000 imagens), o que tornava o treinamento inicial muito lento (o Colab estimou horas, com 582 passos por época).
* **Minha Solução (Prototipagem Rápida):** Para testar minha arquitetura de modelo rapidamente, eu dividi manualmente o dataset usando "slices" (fatias) do TFDS.

Minhas divisões foram:

* **Treinamento:** `train[:15%]` (Os primeiros 15% dos dados)
* **Validação:** `train[15%:20%]` (Os próximos 5%)
* **Teste:** `train[20%:25%]` (Os 5% seguintes)

Isso me deu um conjunto de dados pequeno o suficiente para treinar em minutos, permitindo-me validar minha abordagem antes de escalar.

## 🔬 Minha Metodologia (Pipeline do Código)

Eu estruturei meu código em 6 etapas claras:

### 1. Carregamento dos Dados

* Carreguei o `cats_vs_dogs` do TFDS usando os splits de 15%/5%/5% que defini.
* Usei `as_supervised=True` para carregar os dados no formato `(imagem, label)`.

### 2. Pré-processamento

* Criei uma função para redimensionar as imagens para `(160, 160)`, o tamanho de entrada que o MobileNetV2 espera.
* Apliquei a função `tf.keras.applications.mobilenet_v2.preprocess_input`, que normaliza os pixels para o intervalo `[-1, 1]`.
* Preparei o pipeline de dados com `.shuffle()`, `.batch()` e `.prefetch()` para garantir um treinamento eficiente.

### 3. Criação do Modelo (Feature Extraction)

* Carreguei o MobileNetV2 com `weights='imagenet'` e `include_top=False` (para remover a camada de classificação original).
* "Congelei" o modelo base definindo `base_model.trainable = False`.
* Adicionei minha própria "cabeça" de classificação no topo:
    * `GlobalAveragePooling2D`: Para achatar os mapas de características.
    * `Dropout(0.2)`: Para regularização.
    * `Dense(1, activation='sigmoid')`: Minha camada de saída. Escolhi 1 neurônio com `sigmoid` por ser um problema de classificação binária.

### 4. Compilação e Treinamento (Fase 1)

* Compilei o modelo com `Adam`, perda `binary_crossentropy` e métrica `accuracy`.
* Treinei o modelo por 10 épocas, observando a performance nos dados de validação.

### 5. Ajuste Fino (Fase 2)

* Defini `base_model.trainable = True` para "descongelar" o modelo.
* Decidi re-congelar as primeiras 100 camadas (`fine_tune_at = 100`) para proteger os pesos mais básicos e treinar apenas as camadas mais abstratas.
* Recompilei o modelo com uma taxa de aprendizado 10x menor (`0.00001`).
* Continuei o treinamento por mais 10 épocas.

### 6. Avaliação

* Avaliei o desempenho final no conjunto de teste (dados que o modelo nunca viu durante o treino ou validação).
* Plotei os gráficos de acurácia e perda (Treino vs. Validação) para analisar visualmente o progresso e verificar se houve overfitting.

## 💻 Como Executar

* **Ambiente:** Usei o Google Colab para este estudo.
* **Acelerador:** Habilitei a GPU gratuita (Ambiente de execução -> Alterar tipo de ambiente de execução -> GPU).
* **Execução:** Colei o script `cats_vs_dogs_transfer_15pct.py` em uma célula e executei.

## 📊 O que eu observei

Mesmo treinando em apenas **15%** dos dados, o poder do Transfer Learning foi impressionante. Consegui uma acurácia de validação e teste muito alta (acima de 95%) em pouquíssimo tempo. O fine-tuning ajudou a "polir" o modelo e ganhar mais alguns pontos de precisão.

Este estudo foi uma ótima validação de que é possível desenvolver modelos de visão computacional de alta performance sem dias de treinamento, e me ensinou uma estratégia eficaz para prototipar rapidamente.
