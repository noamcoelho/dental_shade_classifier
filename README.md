# Dental Shade Classifier

Este projeto utiliza visão computacional e deep learning para classificar tons dentais em imagens.

## Estrutura do Projeto

```
dental_shade_classifier/
├── dental_shade_classifier/
│   └── datasets/
│       ├── train/
│       │   ├── A1/
│       │   ├── A2/
│       │   ├── B2/
│       │   └── C1/
│       └── test/
│           ├── A1/
│           ├── A2/
│           ├── B2/
│           └── C1/
├── train_model.py
├── predict_image.py
├── evaluate_model.py
├── confusion_matrix.py
├── dental_shade_classifier_model.keras
└── README.md
```

## Scripts

- **train_model.py**: Treina o modelo de classificação de imagens usando Transfer Learning com ResNet50 pré-treinado no ImageNet e salva o modelo no formato `.keras`.
- **predict_image.py**: Faz a predição da classe de uma imagem individual usando o modelo `.keras`.
- **evaluate_model.py**: Avalia o modelo em todas as imagens de teste, mostra acurácia/confiança e usa o modelo `.keras`.
- **confusion_matrix.py**: Gera e exibe a matriz de confusão das previsões usando o modelo `.keras`.

## Como usar

1. Instale as dependências:
   ```bash
   pip install tensorflow scikit-learn matplotlib numpy
   ```
2. Organize as imagens nas pastas `train` e `test`, separadas por classe.
3. Treine o modelo (usando ResNet50 e salvando em formato moderno):
   ```bash
   python train_model.py
   ```
4. Faça previsões em uma imagem:
   ```bash
   python predict_image.py
   ```
5. Avalie o modelo em lote:
   ```bash
   python evaluate_model.py
   ```
6. Gere a matriz de confusão:
   ```bash
   python confusion_matrix.py
   ```

## Observações
- Certifique-se de que as imagens estejam corretamente organizadas em subpastas por classe.
- O modelo salvo será `dental_shade_classifier_model.keras` (formato recomendado pelo Keras).
- Ajuste os parâmetros dos scripts conforme necessário para seu caso.

### Atualizações recentes
- Agora o projeto utiliza Transfer Learning com ResNet50 pré-treinado.
- O modelo é salvo e carregado no formato `.keras` (mais moderno e recomendado).

---

Projeto para fins educacionais e de pesquisa.
