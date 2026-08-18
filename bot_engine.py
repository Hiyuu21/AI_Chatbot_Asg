import os
import json
import pickle
import joblib
import numpy as np
import torch

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

from new_preprocess import preprocess_text
from transformer import Tokenizer, TransformerModel

class BotEngine:
    def __init__(self):
        self.current_model_type = None
        
        # Artifact placeholders
        self.model = None
        self.tokenizer_or_vectorizer = None
        self.encoder = None
        self.responses = {}
        self.config = {}
        
        # Setup device for Transformer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_model(self, model_name):
        """Loads the corresponding artifacts into memory based on the selected model."""
        print(f"\n[BotEngine] Attempting to load {model_name}...")
        
        try:
            if model_name == "SVM":
                self.tokenizer_or_vectorizer = joblib.load("svm_vectorizer.pkl")
                self.model = joblib.load("trained_svm.pkl")
                
                with open("svm_vocab.json", 'r', encoding="utf-8") as f:
                    data = json.load(f)
                    self.encoder = data['id_to_intent']
                    
                with open("svm_responses.json", "r", encoding="utf-8") as f:
                    self.responses = json.load(f)
                    
            elif model_name == "Transformer":
                with open("transformer_vocab.json", 'r', encoding="utf-8") as f:
                    data = json.load(f)
                    vocab = data['vocab']
                    self.encoder = data['id_to_intent']
                    num_intents = data['num_intents']
                    
                with open("transformer_responses.json", "r", encoding="utf-8") as f:
                    self.responses = json.load(f)

                # Read the architecture back out of the file train_transformer.py
                # saves it to, instead of hardcoding the numbers here. This is the
                # single source of truth now — if the architecture changes (e.g.
                # d_model gets bumped again), this file picks it up automatically
                # instead of needing a manual edit that can drift out of sync.
                with open("transformer_config.json", "r", encoding="utf-8") as f:
                    self.config = json.load(f)

                self.tokenizer_or_vectorizer = Tokenizer()
                self.tokenizer_or_vectorizer.vocab = vocab
                
                self.model = TransformerModel(
                    vocab_size=len(vocab) + 50,
                    num_intents=num_intents,
                    d_model=self.config["d_model"],
                    max_length=self.config["max_length"],
                    ff_hidden_dim=self.config["ff_hidden_dim"],
                    num_heads=self.config["num_heads"],
                    num_blocks=self.config["num_blocks"],
                    embedder_dropout=self.config["embedder_dropout"],
                    ffn_dropout=self.config["ffn_dropout"],
                    classifier_dropout=self.config["classifier_dropout"],
                )
                self.model.load_state_dict(torch.load("trained_transformer.pth", map_location=self.device, weights_only=True))
                self.model.to(self.device)
                self.model.eval()
                
            elif model_name == "LSTM":
                self.model = load_model("lstm_model.keras")
                
                with open("lstm_tokenizer.pkl", "rb") as f:
                    self.tokenizer_or_vectorizer = pickle.load(f)
                    
                with open("lstm_label_encoder.pkl", "rb") as f:
                    self.encoder = pickle.load(f)
                    
                with open("lstm_responses.pkl", "rb") as f:
                    self.responses = pickle.load(f)
                    
                with open("lstm_config.pkl", "rb") as f:
                    self.config = pickle.load(f)

            self.current_model_type = model_name
            print(f"[BotEngine] {model_name} loaded successfully!")
            return True
            
        except Exception as e:
            print(f"[BotEngine] Error loading {model_name}: {e}")
            return False

    def get_response(self, user_text):
        """Processes the text, runs inference on the active model, and applies confidence logic."""
        if not self.model:
            return "Please select a model from the sidebar first.", "NO_MODEL", 0.0
            
        clean_text = preprocess_text(user_text)
        if not clean_text:
            return "I couldn't understand that. Could you please provide more details?", "NO_INPUT", 0.0

        intent = "UNKNOWN"
        confidence = 0.0

        # --- SVM INFERENCE ---
        if self.current_model_type == "SVM":
            input_vector = self.tokenizer_or_vectorizer.transform([clean_text])
            pred_idx = self.model.predict(input_vector)[0]
            probs = self.model.predict_proba(input_vector)[0]
            
            confidence = float(probs[pred_idx])
            intent = self.encoder.get(str(pred_idx), "UNKNOWN")

        # --- TRANSFORMER INFERENCE ---
        elif self.current_model_type == "Transformer":
            token_ids = self.tokenizer_or_vectorizer.encode(clean_text, max_length=self.config["max_length"])
            input_tensor = torch.tensor(token_ids).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                logits = self.model(input_tensor)
                
            probs = torch.softmax(logits, dim=-1)
            conf, pred_idx = torch.max(probs, dim=-1)
            
            confidence = float(conf.item())
            intent = self.encoder.get(str(pred_idx.item()), "UNKNOWN")

        # --- LSTM INFERENCE ---
        elif self.current_model_type == "LSTM":
            seq = self.tokenizer_or_vectorizer.texts_to_sequences([clean_text])
            if len(seq[0]) > 0:
                padded = pad_sequences(seq, maxlen=self.config['max_length'], padding="post", truncating="post")
                prediction = self.model.predict(padded, verbose=0)
                
                confidence = float(np.max(prediction))
                pred_idx = int(np.argmax(prediction))
                intent = self.encoder.inverse_transform([pred_idx])[0]

        # --- CONFIDENCE THRESHOLD LOGIC ---
        if confidence < 0.5:
            reply = "I am not completely sure what you mean. Could you rephrase your question?"
        elif 0.5 <= confidence <= 0.8:
            reply = f"I'm sorry I'm not sure about your question. It seems like you're asking about {intent}, can you please clarify your question?"
        else:
            reply = self.responses.get(intent, "I'm sorry, I don't have a response mapped for that yet.")

        return reply, intent, confidence