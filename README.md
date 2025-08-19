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
├── dental_shade_classifier_model.h5
└── README.md
```

## Scripts

- **train_model.py**: Treina o modelo de classificação de imagens.
- **predict_image.py**: Faz a predição da classe de uma imagem individual.
- **evaluate_model.py**: Avalia o modelo em todas as imagens de teste e mostra acurácia/confiança.
- **confusion_matrix.py**: Gera e exibe a matriz de confusão das previsões.

## Como usar

1. Instale as dependências:
   ```bash
   pip install tensorflow scikit-learn matplotlib numpy
   ```
2. Organize as imagens nas pastas `train` e `test`, separadas por classe.
3. Treine o modelo:
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
- O modelo salvo será `dental_shade_classifier_model.h5`.
- Ajuste os parâmetros dos scripts conforme necessário para seu caso.

---

Projeto para fins educacionais e de pesquisa.
