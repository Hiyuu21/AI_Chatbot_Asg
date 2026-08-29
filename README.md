# TARUMT FAQ Chatbot

## Prerequisites

This project requires several distinct data science and machine learning libraries to support all three NLP architectures (PyTorch, TensorFlow, and Scikit-Learn) alongside the graphical interface.

Run the following command to install all necessary dependencies:

```bash
pip install torch tensorflow scikit-learn pandas numpy matplotlib customtkinter nltk joblib openpyxl

```

(Note: `openpyxl` is strictly required for `pandas` to read the `faq_idk7.xlsx` dataset.)

---

## Project Structure & File Guide

### 1. Data Pipeline

* **`preprocess.py`**: The centralized data handler. It manages dataset loading, applies NLTK-based text normalization (lemmatization, punctuation stripping, abbreviation expansion), and executes the crucial `group_aware_split` to prevent train/test data leakage.



### 2. Model Training Scripts

These scripts are run offline to generate the necessary model artifacts.

* **`train_svm.py`**: Trains the Machine Learning baseline using Scikit-Learn. It fits a `TfidfVectorizer` and a `CalibratedClassifierCV` (wrapping a `LinearSVC`), then exports `.pkl` artifacts.


* **`train_lstm.py`** (Source 6): Trains the sequential Deep Learning model using TensorFlow/Keras. It handles Keras tokenization, sequence padding, and exports a `.keras` model file alongside training history charts.


* **`transformer.py`**: Contains the foundational PyTorch classes for the from-scratch Transformer architecture, including the custom `Tokenizer`, `Embedder`, `MultiHeadAttention`, and the final `TransformerModel`.


* **`train_transformer.py`**: The training loop for the PyTorch Transformer. It includes early-stopping logic, diagnostic metric reporting, and exports the `trained_transformer.pth` weights.



### 3. Deployment & Application

* **`bot_engine.py`**: The backend inference controller. It dynamically routes text through the correct preprocessing steps, loads the selected model's artifacts into memory, executes predictions, and applies the safety confidence-threshold logic.


* **`bot_ui.py`**: The frontend graphical user interface built with CustomTkinter. This is the main entry point for end-users to interact with the models.



---

## Quick Start Guide

1. Ensure the raw dataset (`faq_idk7.xlsx`) is placed in the root directory.

2. The trained models' paths are included in this zipped folder and are ready to use. Run `bot_ui.py` to launch the graphical interface. Select your trained model from the sidebar dropdown to begin chatting.