import pickle
import random
import time
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

from keras_preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.utils import to_categorical

# Import centralized preprocessing functions
from new_preprocess import load_and_clean_data, group_aware_split

# ==========================================================
# CONFIGURATION & RANDOM SEED
# ==========================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

DATASET = "faq_idk7.xlsx"

EMBEDDING_DIM = 128
LSTM_UNITS = 128
DROPOUT_RATE = 0.5
TEST_SIZE = 0.20
EPOCHS = 8
BATCH_SIZE = 16
SEQUENCE_PERCENTILE = 95

# ==========================================================
# LOAD, CLEAN, AND SPLIT DATASET
# ==========================================================
print("=" * 60)
print("Loading and Splitting Dataset...")
print("=" * 60)

raw_data, responses = load_and_clean_data(DATASET)

if not raw_data:
    raise ValueError("Dataset could not be loaded or is empty.")

# Save responses mapping
with open("lstm_responses.pkl", "wb") as f:
    pickle.dump(responses, f)
print("\n✓ lstm_responses.pkl saved.")

# Perform Group-Aware Split
raw_train, raw_test, _ = group_aware_split(raw_data, test_size=TEST_SIZE, seed=SEED)

# Extract texts and labels (Text is already preprocessed by load_and_clean_data)
X_train_text = [row[0] for row in raw_train]
y_train_labels = [row[1] for row in raw_train]

X_test_text = [row[0] for row in raw_test]
y_test_labels = [row[1] for row in raw_test]

# ==========================================================
# LABEL ENCODING
# ==========================================================
encoder = LabelEncoder()

# Fit on all unique intents to ensure all classes are captured
all_intents = sorted(list(set([row[1] for row in raw_data])))
encoder.fit(all_intents)

y_train_integer = encoder.transform(y_train_labels)
y_test_integer = encoder.transform(y_test_labels)

num_classes = len(encoder.classes_)
print(f"\nNumber of Intents mapped: {num_classes}")

with open("lstm_label_encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)
print("✓ lstm_label_encoder.pkl saved.")

# ==========================================================
# CREATE TOKENIZER (Fitted ONLY on Training Data)
# ==========================================================
print("\n" + "=" * 60)
print("Creating Tokenizer...")
print("=" * 60)

tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(X_train_text)
vocab_size = len(tokenizer.word_index) + 1
print(f"Vocabulary Size: {vocab_size}")

with open("lstm_tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)
print("✓ lstm_tokenizer.pkl saved.")

# Convert texts to sequences
train_sequences = tokenizer.texts_to_sequences(X_train_text)
test_sequences = tokenizer.texts_to_sequences(X_test_text)

# Determine maximum sequence length based on percentile
train_lengths = [len(seq) for seq in train_sequences]
max_length = int(np.percentile(train_lengths, SEQUENCE_PERCENTILE))
max_length = max(max_length, 1)

print(f"Selected Maximum Sequence Length: {max_length}")

# Pad sequences
X_train = pad_sequences(train_sequences, maxlen=max_length, padding="post", truncating="post")
X_test = pad_sequences(test_sequences, maxlen=max_length, padding="post", truncating="post")

print(f"Training Input Shape: {X_train.shape}")
print(f"Testing Input Shape: {X_test.shape}")

# One-hot encode targets
y_train = to_categorical(y_train_integer, num_classes=num_classes)
y_test = to_categorical(y_test_integer, num_classes=num_classes)

# ==========================================================
# SAVE CONFIGURATION
# ==========================================================
with open("lstm_config.pkl", "wb") as f:
    pickle.dump({
        "max_length": max_length,
        "vocab_size": vocab_size,
        "embedding_dim": EMBEDDING_DIM,
        "lstm_units": LSTM_UNITS,
        "num_classes": num_classes,
        "sequence_percentile": SEQUENCE_PERCENTILE,
        "test_size": TEST_SIZE,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "dropout_rate": DROPOUT_RATE
    }, f)
print("✓ lstm_config.pkl saved.")

# ==========================================================
# BUILD LSTM MODEL
# ==========================================================
print("\n" + "=" * 60)
print("Building LSTM Model...")
print("=" * 60)

model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=EMBEDDING_DIM, mask_zero=True))
model.add(LSTM(LSTM_UNITS, return_sequences=False))
model.add(Dropout(DROPOUT_RATE))
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(num_classes, activation="softmax"))

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()

# ==========================================================
# TEST ACCURACY CALLBACK
# ==========================================================
class TestAccuracyCallback(tf.keras.callbacks.Callback):
    def __init__(self, X_test, y_test):
        super().__init__()
        self.X_test = X_test
        self.y_test = y_test
        self.test_accuracy = []
        self.test_loss = []

    def on_epoch_end(self, epoch, logs=None):
        loss, accuracy = self.model.evaluate(self.X_test, self.y_test, verbose=0)
        self.test_loss.append(loss)
        self.test_accuracy.append(accuracy)
        print(f" - test_loss: {loss:.4f} - test_accuracy: {accuracy:.4f}")

test_callback = TestAccuracyCallback(X_test, y_test)

# ==========================================================
# TRAIN MODEL
# ==========================================================
print("\n" + "=" * 60)
print("Training LSTM Model...")
print("=" * 60)

start_time = time.time()

history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[test_callback],
    verbose=1
)

end_time = time.time()
print(f"--- LSTM Training Time: {(end_time - start_time) / 60:.4f} minutes ---")

# ==========================================================
# FINAL EVALUATION & SAVING
# ==========================================================
print("\n" + "=" * 60)
print("Final Evaluation on Test Data...")
print("=" * 60)

test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Final Test Loss: {test_loss:.4f}")
print(f"Final Test Accuracy: {test_accuracy * 100:.2f}%\n")

model.save("lstm_model.keras")
print("✓ lstm_model.keras saved.")

# Save Training History
history_df = pd.DataFrame({
    "Epoch": range(1, EPOCHS + 1),
    "Training Accuracy": history.history["accuracy"],
    "Test Accuracy": test_callback.test_accuracy,
    "Training Loss": history.history["loss"],
    "Test Loss": test_callback.test_loss
})
history_df.to_csv("lstm_training_history.csv", index=False)
print("✓ lstm_training_history.csv saved.")

# Generate Graphics
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# --- Loss subplot ---
ax = axes[0]
ax.plot(range(1, EPOCHS + 1), history.history["loss"], label="Training Loss", color="tab:blue", linewidth=2)
ax.plot(range(1, EPOCHS + 1), test_callback.test_loss, label="Test Loss", color="tab:orange", linewidth=2)
ax.set_title("LSTM Model Loss")         # FIXED
ax.set_xlabel("Epoch")                  # FIXED
ax.set_ylabel("Loss")                   # FIXED
ax.set_xticks(range(1, EPOCHS + 1))     # FIXED
ax.set_ylim(bottom=0)
ax.legend()
ax.grid(alpha=0.3)                      # Added alpha=0.3 so both charts look consistent

# --- Accuracy subplot ---
ax = axes[1]
ax.plot(range(1, EPOCHS + 1), history.history["accuracy"], label="Training Accuracy", color="tab:blue", linewidth=2)
ax.plot(range(1, EPOCHS + 1), test_callback.test_accuracy, label="Test Accuracy", color="tab:orange", linewidth=2)
ax.set_title("LSTM Model Accuracy")     # FIXED
ax.set_xlabel("Epoch")                  # FIXED
ax.set_ylabel("Accuracy")               # FIXED
ax.set_xticks(range(1, EPOCHS + 1))     # FIXED
ax.set_ylim(0, 1.05)                    # FIXED to match Keras 0.0 - 1.0 decimal accuracy output
ax.legend()
ax.grid(alpha=0.3)

fig.suptitle("Training Performance Trend (Overfit Check)", fontsize=14)
fig.tight_layout()
fig.savefig("lstm_loss_accuracy.png", dpi=150)
plt.close(fig)
print("✓ lstm_loss_accuracy.png saved.")

# Classification Report & Confusion Matrix
predictions = model.predict(X_test, verbose=0)
predicted_class = np.argmax(predictions, axis=1)

classification_result = classification_report(
    y_test_integer, predicted_class,
    labels=np.arange(num_classes),
    target_names=encoder.classes_,
    zero_division=0,
    digits=4
)

with open("lstm_classification_report.txt", "w", encoding="utf-8") as f:
    f.write(classification_result)
print("✓ lstm_classification_report.txt saved.")

cm = confusion_matrix(y_test_integer, predicted_class, labels=np.arange(num_classes))
cm_df = pd.DataFrame(cm, index=encoder.classes_, columns=encoder.classes_)
cm_df.to_csv("lstm_confusion_matrix.csv")
print("✓ lstm_confusion_matrix.csv saved.")

print("\n" + "=" * 60)
print("LSTM Training Completed Successfully.")
print("=" * 60)