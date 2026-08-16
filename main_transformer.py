import torch
import json
import os

from preprocess import preprocess_text
from transformer import Tokenizer, TransformerModel

def load_faq_bot():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"-- Booting up. Running on {device}")

    if not os.path.exists("vocab.json") or not os.path.exists("responses.json"):
        print("Error: 'vocab.json' not found. Complete training cycle first.")
        return None, None, None, None, None

    with open("vocab.json", 'r', encoding="utf-8") as f:
        config_data = json.load(f)

    with open("responses.json", "r", encoding="utf-8") as f:
        intent_responses = json.load(f)

    tokenizer = Tokenizer()
    tokenizer.vocab = config_data["vocab"]
    num_intents = config_data['num_intents']
    id_to_intent = config_data['id_to_intent']

    print(f"--> Loaded vocabulary with {len(tokenizer.vocab)} words.")
    print(f"--> Mapped {num_intents} corporate intents.")

    MAX_LENGTH = 20
    D_MODEL = 64
    FF_HIDDEN_DIM = 128

    model = TransformerModel(
        vocab_size=len(tokenizer.vocab) + 50,
        num_intents=num_intents,
        d_model=D_MODEL,
        max_length=MAX_LENGTH,
        ff_hidden_dim=FF_HIDDEN_DIM 
    )

    if not os.path.exists("trained_transformer.pth"):
        print("ERROR: 'trained_transformer.pth' weights file missing. Complete training cycle first.")
        return None, None, None, None, None
        
    model.load_state_dict(torch.load("trained_transformer.pth", map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return tokenizer, model, id_to_intent, intent_responses, device

def start_chat_session():
    tokenizer, model, id_to_intent, intent_responses, device = load_faq_bot()
    if model is None:
        return
        
    print("\n" + "="*60)
    print("-- System Ready. Type 'quit' or 'exit' to close the session.")
    print("="*60 + "\n")
    
    MAX_LENGTH = 20
    
    while True:
        try:
            user_input = input("\nYou: ")
            if user_input.strip().lower() in ['quit', 'exit']:
                print("Bot: Session terminated. Goodbye!")
                break
                
            if not user_input.strip():
                continue
                
            # Execute preprocessing
            clean_text = preprocess_text(user_input)
            
            # Convert sequence to numerical matrix representation
            token_ids = tokenizer.encode(clean_text, max_length=MAX_LENGTH)
            
            # Reshape sequence vector into dimension batch matching [1, sequence_length]
            input_tensor = torch.tensor(token_ids).unsqueeze(0).to(device)
            
            # Run forward evaluation execution pass without gradients
            with torch.no_grad():
                logits = model(input_tensor)
                
            # Convert evaluation logits scores to localized distributions
            probabilities = torch.softmax(logits, dim=-1)
            confidence, predicted_class_idx = torch.max(probabilities, dim=-1)
            
            # Format outputs back out of evaluation environments
            confidence_score = confidence.item()
            predicted_id_str = str(predicted_class_idx.item())
            
            # Interpret predictions using validation ledger mappings
            detected_intent = id_to_intent.get(predicted_id_str, "UNKNOWN_INTENT")
            
            # Apply strict structural classification guide and print the mapped response
            if confidence_score < 0.5:
                print(f"Bot: I am not completely sure what you mean. Could you rephrase that query?")
                print(f"[Debug info - Highest Guess: '{detected_intent}' with Confidence: {confidence_score:.2%}]")
            elif 0.5 <= confidence_score <= 0.8:
                print(f"Bot: I'm sorry I'm not sure about your question. It seems like you're asking about '{detected_intent}', can you please clarify your question?")
            else:
                # Retrieve the actual response text associated with the detected intent
                bot_reply = intent_responses.get(detected_intent, "I'm sorry, I don't have a response mapped for that yet.")
                print(f"Bot: {bot_reply} - {confidence_score:.2%}")
                
        except Exception as e:
            print(f"Runtime Error Process Interruption: {e}\n")

if __name__ == "__main__":
    start_chat_session()