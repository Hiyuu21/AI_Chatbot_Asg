import json
import os
import joblib

from preprocess import preprocess_text

def load_faq_bot():
    print("-- Booting up. (SVM inference runs on CPU, no GPU required)")

    required_files = ["svm_vectorizer.pkl", "trained_svm.pkl", "svm_vocab.json", "responses.json"]
    for fname in required_files:
        if not os.path.exists(fname):
            print(f"Error: '{fname}' not found. Complete training cycle first.")
            return None, None, None, None

    vectorizer = joblib.load("svm_vectorizer.pkl")
    model = joblib.load("trained_svm.pkl")

    with open("svm_vocab.json", 'r', encoding="utf-8") as f:
        config_data = json.load(f)

    with open("responses.json", "r", encoding="utf-8") as f:
        intent_responses = json.load(f)

    num_intents = config_data['num_intents']
    id_to_intent = config_data['id_to_intent']

    print(f"--> Loaded TF-IDF vectorizer with {len(vectorizer.vocabulary_)} words.")
    print(f"--> Mapped {num_intents} corporate intents.")

    return vectorizer, model, id_to_intent, intent_responses

def start_chat_session():
    vectorizer, model, id_to_intent, intent_responses = load_faq_bot()
    if model is None:
        return

    print("\n" + "="*60)
    print("-- System Ready. Type 'quit' or 'exit' to close the session.")
    print("="*60 + "\n")

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

            # Convert cleaned text into a weighted numerical representation (TF-IDF)
            input_vector = vectorizer.transform([clean_text])

            # Run prediction
            predicted_class_idx = model.predict(input_vector)[0]

            # Convert decision scores to a probability-like confidence value
            probabilities = model.predict_proba(input_vector)[0]
            confidence_score = probabilities[predicted_class_idx]

            # Format outputs
            predicted_id_str = str(predicted_class_idx)

            # Interpret predictions using validation ledger mappings
            detected_intent = id_to_intent.get(predicted_id_str, "UNKNOWN_INTENT")

            # Apply strict structural classification guide and print the mapped response
            if confidence_score < 0.6:
                print(f"Bot: I am not completely sure what you mean. Could you rephrase that query?")
                print(f"[Debug info - Highest Guess: '{detected_intent}' with Confidence: {confidence_score:.2%}]")
            else:
                # Retrieve the actual response text associated with the detected intent
                bot_reply = intent_responses.get(detected_intent, "I'm sorry, I don't have a response mapped for that yet.")
                print(f"Bot: {bot_reply} - {confidence_score:.2%}")

        except Exception as e:
            print(f"Runtime Error Process Interruption: {e}\n")

if __name__ == "__main__":
    start_chat_session()