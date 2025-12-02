# Student Performance — Regression Comparison

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![Jupyter](https://img.shields.io/badge/Notebook-Jupyter-yellow.svg)](https://jupyter.org/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Welberth77/Projeto-IA/blob/main/student-performance-repo/notebooks/analysis.ipynb)

Este repositório implementa e compara modelos de **Regressão Linear** e **Regressão Logística** para previsão de desempenho acadêmico utilizando o dataset *Student Performance* do UCI Machine Learning Repository.

---

##  Objetivo do Projeto

Comparar, utilizando o mesmo conjunto de dados:

| Modelo              | Tipo de Problema | Previsão            |
| ------------------- | ---------------- | ------------------- |
| Linear Regression   | Regressão        | Nota final (G3)     |
| Logistic Regression | Classificação    | Pass/Fail (G3 ≥ 10) |

O código realiza:

* Pré-processamento
* Treinamento
* Avaliação com métricas específicas
* Salvamento dos modelos

---

##  Estrutura do Repositório

```
student-performance-repo/
│
├── src/
│   ├── train_models.py        # Treinamento e avaliação dos modelos
│   └── download_data.py       # Baixa o dataset automaticamente
│
├── notebooks/
│   └── analysis.ipynb         # Notebook com análise exploratória
│
├── models/                    # Modelos salvos após treino
├── data/                      # Local previsto do dataset
│
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

---

##  Dataset

Dataset utilizado:

> **Student Performance Data Set (UCI Machine Learning Repository)**
> [https://archive.ics.uci.edu/ml/datasets/Student+Performance](https://archive.ics.uci.edu/ml/datasets/Student+Performance)

Copie para:

```
data/student-mat.csv
```

Ou baixe automaticamente:

```bash
python src/download_data.py --out data/student-mat.csv
```

---

##  Como Executar o Projeto

### 1. Criar ambiente virtual

```bash
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows
```

### 2. Instalar dependências

```bash
pip install -r requirements.txt
```

### 3. Executar o treinamento

```bash
python src/train_models.py --data data/student-mat.csv --output models
```

O script irá:

* Treinar ambos os modelos
* Mostrar métricas (R², RMSE, Acurácia, AUC etc.)
* Salvar modelos em `models/`

### 4. Executar o notebook (opcional)

```bash
jupyter notebook notebooks/analysis.ipynb
```

---

##  Métricas Calculadas

Para Regressão Linear:

* R²
* RMSE

Para Regressão Logística:

* Accuracy
* ROC-AUC
* Matriz de Confusão
* Classification Report

---

##  Reprodutibilidade

Todos os scripts são determinísticos:

* random_state fixado
* divisão treino/teste consistente

---

##  Como enviar para o GitHub

```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/Welberth77/Projeto-IA.git
git push -u origin main
```

---

##  Dependências Principais

```
numpy
pandas
scikit-learn
matplotlib
joblib
jupyter
```

## Link do Google Colab


> [https://colab.research.google.com/github/Welberth77/Projeto-IA/blob/main/student-performance-repo/notebooks/analysis.ipynb](https://colab.research.google.com/github/Welberth77/Projeto-IA/blob/main/student-performance-repo/notebooks/analysis.ipynb)

---

## Licença

Este projeto é distribuído sob a licença **MIT**.
