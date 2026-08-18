from new_preprocess import load_and_clean_data, group_aware_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report
import json
import time
import joblib


"""
SVM model train
"""

def train_svm():
    raw_data, intent_to_response = load_and_clean_data("faq_idk7.xlsx")
    if not raw_data: return

    with open("svm_responses.json", "w", encoding="utf-8") as f:
        json.dump(intent_to_response, f, indent=4)

    raw_train, raw_test, _ = group_aware_split(raw_data, test_size=0.20, seed=42)

    # Extract all unique intents and sort them alphabetically for consistency
    # (same ordering rule as the Transformer stream, so intent IDs line up across models)
    unique_intents = sorted(list(set([row[1] for row in raw_data])))
    intent_to_id = {intent: idx for idx, intent in enumerate(unique_intents)}
    id_to_intent = {idx: intent for idx, intent in enumerate(unique_intents)}
    num_intents = len(unique_intents)
    print(f"-- Successfully mapped {num_intents} unique intents.")

    # Execute preprocessing (identical cleaning step used by all 3 models)
    train_sentences = [row[0] for row in raw_train]
    train_labels = [intent_to_id[row[1]] for row in raw_train]

    test_sentences = [row[0] for row in raw_test]
    test_labels = [intent_to_id[row[1]] for row in raw_test]

    # ---------------------------------------------------------------
    # TF-IDF vectorization (SVM-specific formatting path)
    # Fit only on the training split (same principle as tokenizer.build_vocab()
    # only seeing train_sentences in evaluation.py) to avoid test-set leakage.
    # ---------------------------------------------------------------
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(train_sentences)
    X_test = vectorizer.transform(test_sentences)

    print(f"-- TF-IDF vocabulary size: {len(vectorizer.vocabulary_)}")

    # ---------------------------------------------------------------
    # Train the SVM classifier
    # probability=True gives confidence scores at inference time,
    # the same role softmax plays for the Transformer's logits.
    # ---------------------------------------------------------------
    #model = SVC(kernel="linear", probability=True, random_state=42)
    print("Starting SVM training...")
    start_time = time.time()
    model = CalibratedClassifierCV(LinearSVC(random_state=42, dual=False), ensemble=False)
    model.fit(X_train, train_labels)
    end_time = time.time()
    print(f"--- SVM Training Time: {(end_time - start_time):.4f} seconds ---")

    # Evaluate on held-out test split (same metrics reported in evaluation.py)
    train_preds = model.predict(X_train)
    train_accuracy = accuracy_score(train_labels, train_preds) * 100

    train_precision, train_recall, train_f1, _ = precision_recall_fscore_support(
        train_labels,
        train_preds,
        average='weighted',
        zero_division=0 
    )

    test_preds = model.predict(X_test)
    test_accuracy = accuracy_score(test_labels, test_preds) * 100

    precision, recall, f1, _ = precision_recall_fscore_support(
        test_labels,
        test_preds,
        average='weighted',
        zero_division=0  # Prevents warnings if a class is never predicted
    )

    print("\nTrain ---")
    print(f"\n--> Accuracy:  {train_accuracy:.4f}%", end=" ")
    print(f"--> Precision: {train_precision:.4f}")
    print(f"--> Recall:    {train_recall:.4f}", end=" ")
    print(f"--> F1-Score:  {train_f1:.4f}")
    print('-'*50 + "\n")


    print("\nTest ---")
    print(f"\n--> Accuracy:  {test_accuracy:.4f}%", end=" ")
    print(f"--> Precision: {precision:.4f}")
    print(f"--> Recall:    {recall:.4f}", end=" ")
    print(f"--> F1-Score:  {f1:.4f}")
    print('-'*50 + "\n")

    # ---------------------------------------------------------------
    # Generate and save the detailed classification report
    # ---------------------------------------------------------------
    target_names = [id_to_intent[i] for i in range(num_intents)]
    report = classification_report(
        test_labels, 
        test_preds, 
        labels=list(range(num_intents)),
        target_names=target_names, 
        zero_division=0,
        digits=4
    )
    
    with open("svm_classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report)
    print("svm_classification_report.txt saved.")

    # ---------------------------------------------------------------
    # Save artifacts (analogous to vocab.json / trained_bot.pth)
    # ---------------------------------------------------------------
    joblib.dump(vectorizer, "svm_vectorizer.pkl")
    joblib.dump(model, "trained_svm.pkl")

    with open("svm_vocab.json", 'w', encoding="utf-8") as f:
        json.dump({
            "num_intents": num_intents,
            "intent_to_id": intent_to_id,
            "id_to_intent": id_to_intent
        }, f, indent=4)
    print("-- Vocabulary and intent mappings saved as 'svm_vocab.json'.")

    print("\nTraining Complete.")
    print("Saved 'svm_vectorizer.pkl' and 'trained_svm.pkl'.")

if __name__ == "__main__":
    train_svm()
