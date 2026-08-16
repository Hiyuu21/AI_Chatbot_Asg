import os
import json
import pickle
import numpy as np
import pandas as pd
import torch

# Keras / TensorFlow imports
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

# Scikit-Learn imports
import joblib

# Shared imports & Custom Transformer Architecture
from new_preprocess import preprocess_text
from transformer import Tokenizer, TransformerModel

# ==========================================================
# 1. EVALUATION QUESTIONS (20 Overlapping Core Intents)
# ==========================================================
QUESTIONS = [
    "Could you list out the bachelor degree options available at the Penang branch?",
    "I'm doing a Diploma in Accounting, which faculty am I actually under?",
    "What foundation courses do you guys offer before jumping into a degree?",
    "Where do I grab the latest TAR UMT prospectus for this year?",
    "If I realize my current course isn't for me, can I easily swap to another one within the same faculty?",
    "I'm an international applicant, can I just use my passport number instead of an NRIC for the application?",
    "Is there a checklist of things I absolutely need to bring for orientation week?",
    "What's the procedure to lock in my co-curriculum activities for the semester?",
    "I need step-by-step instructions on applying for the PTPTN loan, where do I start?",
    "When is the actual deadline for me to start paying back my PTPTN loan after I graduate?",
    "Money is a bit tight right now, is there a way to pay my fees in installments?",
    "Will I get penalized or kicked out of class if my tuition fee payment is a few days late?",
    "I got pretty good results, how can I apply for a merit scholarship?",
    "Does the Penang campus provide its own hostels, or do I need to find my own place outside?",
    "Where can I check the departure times for the campus shuttle bus?",
    "Are there any good, quiet spots on campus to study for my finals besides the library?",
    "Where are the student printing services located, and how do I pay for them?",
    "What's the maximum number of books I can borrow from the library at once?",
    "I totally forgot my Intranet login, how do I go about resetting the password?",
    "My laptop can't connect to the campus Wi-Fi, where exactly is the IT helpdesk located so I can get help?"
]

# ==========================================================
# 2. MODEL LOADERS & INFERENCE WRAPPERS
# ==========================================================

# ----------------------------------------------------------
# A. LSTM Model Loader & Predictor
# ----------------------------------------------------------
def get_lstm_predictor():
    required_files = ["lstm_model.keras", "lstm_tokenizer.pkl", "lstm_label_encoder.pkl", "lstm_responses.pkl", "lstm_config.pkl"]
    for fname in required_files:
        if not os.path.exists(fname):
            print(f"[LSTM] Missing file: {fname}. Skipping LSTM evaluation.")
            return None

    model = load_model("lstm_model.keras")
    with open("lstm_tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    with open("lstm_label_encoder.pkl", "rb") as f:
        encoder = pickle.load(f)
    with open("lstm_responses.pkl", "rb") as f:
        responses = pickle.load(f)
    with open("lstm_config.pkl", "rb") as f:
        config = pickle.load(f)

    max_length = config["max_length"]

    def predict(raw_question):
        clean_text = preprocess_text(raw_question)
        if not clean_text:
            return "NO_INPUT", 0.0, "Empty input."

        seq = tokenizer.texts_to_sequences([clean_text])
        if len(seq[0]) == 0:
            return "UNKNOWN", 0.0, "No vocabulary match."

        padded = pad_sequences(seq, maxlen=max_length, padding="post", truncating="post")
        prediction = model.predict(padded, verbose=0)
        
        confidence = float(np.max(prediction))
        predicted_index = int(np.argmax(prediction))
        intent = encoder.inverse_transform([predicted_index])[0]
        response = responses.get(intent, "No response mapped.")

        return intent, confidence, response

    return predict


# ----------------------------------------------------------
# B. SVM Model Loader & Predictor
# ----------------------------------------------------------
def get_svm_predictor():
    required_files = ["svm_vectorizer.pkl", "trained_svm.pkl", "svm_vocab.json", "svm_responses.json"]
    for fname in required_files:
        if not os.path.exists(fname):
            print(f"[SVM] Missing file: {fname}. Skipping SVM evaluation.")
            return None

    vectorizer = joblib.load("svm_vectorizer.pkl")
    model = joblib.load("trained_svm.pkl")
    
    with open("svm_vocab.json", "r", encoding="utf-8") as f:
        vocab_config = json.load(f)
    with open("svm_responses.json", "r", encoding="utf-8") as f:
        responses = json.load(f)

    id_to_intent = vocab_config["id_to_intent"]

    def predict(raw_question):
        clean_text = preprocess_text(raw_question)
        if not clean_text:
            return "NO_INPUT", 0.0, "Empty input."

        input_vec = vectorizer.transform([clean_text])
        predicted_idx = model.predict(input_vec)[0]
        probabilities = model.predict_proba(input_vec)[0]
        confidence = float(probabilities[predicted_idx])

        detected_intent = id_to_intent.get(str(predicted_idx), "UNKNOWN_INTENT")
        response = responses.get(detected_intent, "No response mapped.")

        return detected_intent, confidence, response

    return predict


# ----------------------------------------------------------
# C. Transformer Model Loader & Predictor
# ----------------------------------------------------------
def get_transformer_predictor():
    required_files = ["transformer_vocab.json", "transformer_responses.json", "trained_transformer.pth"]
    for fname in required_files:
        if not os.path.exists(fname):
            print(f"[Transformer] Missing file: {fname}. Skipping Transformer evaluation.")
            return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open("transformer_vocab.json", "r", encoding="utf-8") as f:
        config_data = json.load(f)
    with open("transformer_responses.json", "r", encoding="utf-8") as f:
        responses = json.load(f)

    tokenizer = Tokenizer()
    tokenizer.vocab = config_data["vocab"]
    num_intents = config_data["num_intents"]
    id_to_intent = config_data["id_to_intent"]

    MAX_LENGTH = 20
    D_MODEL = 32
    FF_HIDDEN_DIM = 64

    model = TransformerModel(
        vocab_size=len(tokenizer.vocab) + 50,
        num_intents=num_intents,
        d_model=D_MODEL,
        max_length=MAX_LENGTH,
        ff_hidden_dim=FF_HIDDEN_DIM
    )

    model.load_state_dict(torch.load("trained_transformer.pth", map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    def predict(raw_question):
        clean_text = preprocess_text(raw_question)
        if not clean_text:
            return "NO_INPUT", 0.0, "Empty input."

        token_ids = tokenizer.encode(clean_text, max_length=MAX_LENGTH)
        input_tensor = torch.tensor(token_ids).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(input_tensor)
            probabilities = torch.softmax(logits, dim=-1)
            confidence, predicted_idx = torch.max(probabilities, dim=-1)

        confidence_score = float(confidence.item())
        detected_intent = id_to_intent.get(str(predicted_idx.item()), "UNKNOWN_INTENT")
        response = responses.get(detected_intent, "No response mapped.")

        return detected_intent, confidence_score, response

    return predict


# ==========================================================
# 3. BENCHMARK EXECUTION & REPORT GENERATION
# ==========================================================
def run_benchmark():
    print("=" * 80)
    print("STARTING AUTOMATED MODEL BENCHMARKING")
    print("=" * 80)

    lstm_predict = get_lstm_predictor()
    svm_predict = get_svm_predictor()
    tf_predict = get_transformer_predictor()

    output_lines = []
    output_lines.append("=" * 100)
    output_lines.append("AUTOMATED CHATBOT MODEL COMPARISON BENCHMARK REPORT")
    output_lines.append("=" * 100)

    for i, question in enumerate(QUESTIONS, start=1):
        clean_q = preprocess_text(question)
        
        header = f"\nQUESTION {i:02d}: \"{question}\"\nPreprocessed: \"{clean_q}\""
        print(header)
        output_lines.append(header)
        output_lines.append("-" * 100)
        output_lines.append(f"{'Model':<15} | {'Confidence':<12} | {'Predicted Intent':<35} | {'Response Preview'}")
        output_lines.append("-" * 100)

        # Run LSTM
        if lstm_predict:
            intent, conf, resp = lstm_predict(question)
            preview = (resp[:40] + "...") if len(resp) > 40 else resp
            row = f"{'LSTM':<15} | {conf:<12.2%} | {intent:<35} | {preview}"
            print(row)
            output_lines.append(row)

        # Run SVM
        if svm_predict:
            intent, conf, resp = svm_predict(question)
            preview = (resp[:40] + "...") if len(resp) > 40 else resp
            row = f"{'SVM':<15} | {conf:<12.2%} | {intent:<35} | {preview}"
            print(row)
            output_lines.append(row)

        # Run Transformer
        if tf_predict:
            intent, conf, resp = tf_predict(question)
            preview = (resp[:40] + "...") if len(resp) > 40 else resp
            row = f"{'Transformer':<15} | {conf:<12.2%} | {intent:<35} | {preview}"
            print(row)
            output_lines.append(row)

        output_lines.append("\n")

    # Save to file
    output_filepath = "model_comparison_results.txt"
    with open(output_filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))

    print("\n" + "=" * 80)
    print(f"BENCHMARK COMPLETE. Results saved to '{output_filepath}'.")
    print("=" * 80)

if __name__ == "__main__":
    run_benchmark()