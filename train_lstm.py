import os
import pickle
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import (
    classification_report,
    confusion_matrix
)

from keras_preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential

from keras.layers import (
    Embedding,
    LSTM,
    Dense,
    Dropout
)

from keras.utils import to_categorical

from preprocess import preprocess_text


# ==========================================================
# RANDOM SEED
# ==========================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ==========================================================
# DATASET
# ==========================================================

DATASET = "faq_idk7.xlsx"


# ==========================================================
# LSTM CONFIGURATION
# ==========================================================

EMBEDDING_DIM = 128

LSTM_UNITS = 128

DROPOUT_RATE = 0.5

TEST_SIZE = 0.20

EPOCHS = 10

BATCH_SIZE = 16

SEQUENCE_PERCENTILE = 95


# ==========================================================
# LOAD DATASET
# ==========================================================

print("=" * 60)
print("Loading Dataset...")
print("=" * 60)


if not os.path.exists(DATASET):

    raise FileNotFoundError(
        f"Dataset not found: {DATASET}"
    )


df = pd.read_excel(DATASET)


print("\nFirst 5 Rows:")
print(df.head())


print("\nDataset Shape:")
print(df.shape)


# ==========================================================
# CHECK REQUIRED COLUMNS
# ==========================================================

required_columns = [
    "Question",
    "Intent",
    "Response"
]


for column in required_columns:

    if column not in df.columns:

        raise Exception(
            f"Missing column: {column}"
        )


print(
    "\nRequired columns found successfully."
)


# ==========================================================
# REMOVE EMPTY ROWS
# ==========================================================

df = df.dropna(
    subset=[
        "Question",
        "Intent",
        "Response"
    ]
)


df = df.reset_index(
    drop=True
)


print(
    "\nAfter Removing Empty Rows:"
)

print(
    df.shape
)


# ==========================================================
# REMOVE DUPLICATE QUESTIONS
# ==========================================================

df["Question"] = df[
    "Question"
].astype(str)


before_duplicates = len(df)


df = df.drop_duplicates(
    subset=["Question"],
    keep="first"
)


df = df.reset_index(
    drop=True
)


duplicates_removed = (
    before_duplicates
    - len(df)
)


print(
    "\nDuplicate Questions Removed:",
    duplicates_removed
)


# ==========================================================
# SAVE ORIGINAL QUESTIONS
# ==========================================================

df["Original_Question"] = df[
    "Question"
]


# ==========================================================
# TEXT PREPROCESSING
# ==========================================================

print("\n" + "=" * 60)
print("Text Preprocessing...")
print("=" * 60)


df["Question"] = df[
    "Question"
].apply(
    preprocess_text
)


# ==========================================================
# REMOVE EMPTY QUESTIONS
# ==========================================================

df = df[
    df["Question"].str.strip() != ""
]


df = df.reset_index(
    drop=True
)


print(
    "\nPreprocessing Completed."
)


# ==========================================================
# DISPLAY PREPROCESSING EXAMPLES
# ==========================================================

print("\n" + "=" * 60)
print("Preprocessing Examples")
print("=" * 60)


sample_count = min(
    10,
    len(df)
)


for i in range(sample_count):

    print(
        f"\nExample {i + 1}"
    )

    print(
        "Original  :",
        df.loc[
            i,
            "Original_Question"
        ]
    )

    print(
        "Processed :",
        df.loc[
            i,
            "Question"
        ]
    )


# ==========================================================
# DATASET STATISTICS
# ==========================================================

print("\n" + "=" * 60)
print("Dataset Statistics")
print("=" * 60)


print(
    "\nNumber of Questions:",
    len(df)
)


print(
    "Number of Intents:",
    df["Intent"].nunique()
)


print(
    "\nIntent Distribution:"
)


print(
    df["Intent"].value_counts()
)


# ==========================================================
# PREPARE QUESTIONS
# ==========================================================

questions = df[
    "Question"
].astype(str)


# ==========================================================
# PREPARE LABELS
# ==========================================================

labels = df[
    "Intent"
].astype(str)


# ==========================================================
# SAVE RESPONSE DICTIONARY
# ==========================================================

responses = {}


for _, row in df.iterrows():

    intent = row[
        "Intent"
    ]

    response = row[
        "Response"
    ]

    if intent not in responses:

        responses[intent] = response


with open(
    "lstm_responses.pkl",
    "wb"
) as f:

    pickle.dump(
        responses,
        f
    )


print(
    "\nlstm_responses.pkl saved."
)


# ==========================================================
# LABEL ENCODING
# ==========================================================

encoder = LabelEncoder()


y_integer = encoder.fit_transform(
    labels
)


num_classes = len(
    encoder.classes_
)


print(
    "\nNumber of Intents:",
    num_classes
)


# ==========================================================
# CHECK MINIMUM CLASS SIZE
# ==========================================================

class_counts = pd.Series(
    y_integer
).value_counts()


if class_counts.min() < 2:

    print("\nWARNING!")

    print(
        "Some intents contain fewer than 2 examples."
    )

    print(
        "Stratified splitting may fail."
    )

    raise ValueError(
        "Each intent must contain at least "
        "2 examples for stratified splitting."
    )


# ==========================================================
# TRAIN / TEST SPLIT
# ==========================================================

print("\n" + "=" * 60)
print("Splitting Dataset...")
print("=" * 60)


# ----------------------------------------------------------
# 80% Training
# 20% Testing
# ----------------------------------------------------------

X_train_text, X_test_text, y_train_integer, y_test_integer = (

    train_test_split(

        questions,

        y_integer,

        test_size=TEST_SIZE,

        random_state=SEED,

        stratify=y_integer
    )
)


print(
    "\nTraining Samples   :",
    len(X_train_text)
)


print(
    "Testing Samples    :",
    len(X_test_text)
)


print(
    "\nTraining Percentage:",
    f"{len(X_train_text) / len(df) * 100:.2f}%"
)


print(
    "Testing Percentage :",
    f"{len(X_test_text) / len(df) * 100:.2f}%"
)


# ==========================================================
# CREATE TOKENIZER
# ==========================================================

print("\n" + "=" * 60)
print("Creating Tokenizer...")
print("=" * 60)


tokenizer = Tokenizer(
    oov_token="<OOV>"
)


# ----------------------------------------------------------
# IMPORTANT:
#
# Tokenizer is fitted ONLY on training data.
#
# This prevents information from the test dataset
# from leaking into the tokenizer vocabulary.
# ----------------------------------------------------------

tokenizer.fit_on_texts(
    X_train_text
)


word_index = tokenizer.word_index


vocab_size = (
    len(word_index)
    + 1
)


print(
    "Vocabulary Size:",
    vocab_size
)


# ==========================================================
# CONVERT TEXT TO SEQUENCES
# ==========================================================

train_sequences = (

    tokenizer.texts_to_sequences(
        X_train_text
    )

)


test_sequences = (

    tokenizer.texts_to_sequences(
        X_test_text
    )

)


# ==========================================================
# DETERMINE MAXIMUM SEQUENCE LENGTH
# ==========================================================

train_lengths = [

    len(sequence)

    for sequence
    in train_sequences

]


print(
    "\nAverage Training Sequence Length:",
    round(
        np.mean(train_lengths),
        2
    )
)


print(
    "Median Training Sequence Length:",
    round(
        np.median(train_lengths),
        2
    )
)


print(
    "Maximum Training Sequence Length:",
    max(train_lengths)
)


# ----------------------------------------------------------
# Use 95th percentile
#
# This prevents one extremely long question from
# determining the padding length for the entire dataset.
# ----------------------------------------------------------

max_length = int(

    np.percentile(

        train_lengths,

        SEQUENCE_PERCENTILE

    )

)


max_length = max(
    max_length,
    1
)


print(
    "Selected Maximum Sequence Length:",
    max_length
)


# ==========================================================
# PAD SEQUENCES
# ==========================================================

X_train = pad_sequences(

    train_sequences,

    maxlen=max_length,

    padding="post",

    truncating="post"

)


X_test = pad_sequences(

    test_sequences,

    maxlen=max_length,

    padding="post",

    truncating="post"

)


print(
    "\nTraining Input Shape:",
    X_train.shape
)


print(
    "Testing Input Shape:",
    X_test.shape
)


# ==========================================================
# SAVE TOKENIZER
# ==========================================================

with open(
    "lstm_tokenizer.pkl",
    "wb"
) as f:

    pickle.dump(
        tokenizer,
        f
    )


print(
    "\nlstm_tokenizer.pkl saved."
)


# ==========================================================
# SAVE LABEL ENCODER
# ==========================================================

with open(
    "lstm_label_encoder.pkl",
    "wb"
) as f:

    pickle.dump(
        encoder,
        f
    )


print(
    "lstm_label_encoder.pkl saved."
)


# ==========================================================
# ONE-HOT ENCODING
# ==========================================================

y_train = to_categorical(

    y_train_integer,

    num_classes=num_classes

)


y_test = to_categorical(

    y_test_integer,

    num_classes=num_classes

)


print(
    "\nTraining Output Shape:",
    y_train.shape
)


print(
    "Testing Output Shape:",
    y_test.shape
)


# ==========================================================
# BUILD LSTM MODEL
# ==========================================================

print("\n" + "=" * 60)
print("Building LSTM Model...")
print("=" * 60)


model = Sequential()


# ==========================================================
# EMBEDDING LAYER
# ==========================================================

model.add(

    Embedding(

        input_dim=vocab_size,

        output_dim=EMBEDDING_DIM,

        mask_zero=True

    )

)


# ==========================================================
# LSTM LAYER
# ==========================================================

model.add(

    LSTM(

        LSTM_UNITS,

        return_sequences=False

    )

)


# ==========================================================
# DROPOUT
# ==========================================================

model.add(

    Dropout(

        DROPOUT_RATE

    )

)


# ==========================================================
# DENSE LAYER
# ==========================================================

model.add(

    Dense(

        64,

        activation="relu"

    )

)


# ==========================================================
# SECOND DROPOUT
# ==========================================================

model.add(

    Dropout(

        0.3

    )

)


# ==========================================================
# OUTPUT LAYER
# ==========================================================

model.add(

    Dense(

        num_classes,

        activation="softmax"

    )

)


# ==========================================================
# COMPILE MODEL
# ==========================================================

model.compile(

    optimizer="adam",

    loss="categorical_crossentropy",

    metrics=["accuracy"]

)


# ==========================================================
# MODEL SUMMARY
# ==========================================================

model.summary()


# ==========================================================
# TEST ACCURACY CALLBACK
# ==========================================================

class TestAccuracyCallback(
    tf.keras.callbacks.Callback
):

    def __init__(
        self,
        X_test,
        y_test
    ):

        super().__init__()

        self.X_test = X_test

        self.y_test = y_test

        self.test_accuracy = []

        self.test_loss = []


    def on_epoch_end(
        self,
        epoch,
        logs=None
    ):

        loss, accuracy = self.model.evaluate(

            self.X_test,

            self.y_test,

            verbose=0

        )


        self.test_loss.append(
            loss
        )


        self.test_accuracy.append(
            accuracy
        )


        print(
            f" - test_loss: {loss:.4f}"
            f" - test_accuracy: {accuracy:.4f}"
        )


# ==========================================================
# CREATE TEST CALLBACK
# ==========================================================

test_callback = TestAccuracyCallback(

    X_test,

    y_test

)


# ==========================================================
# TRAIN MODEL
# ==========================================================

print("\n" + "=" * 60)
print("Training LSTM Model...")
print("=" * 60)


history = model.fit(

    X_train,

    y_train,

    epochs=EPOCHS,

    batch_size=BATCH_SIZE,

    callbacks=[
        test_callback
    ],

    verbose=1

)


# ==========================================================
# FINAL EVALUATION ON TEST DATA
# ==========================================================

print("\n" + "=" * 60)
print("Final Evaluation on Test Data...")
print("=" * 60)


test_loss, test_accuracy = model.evaluate(

    X_test,

    y_test,

    verbose=0

)


print(
    f"\nFinal Test Loss     : "
    f"{test_loss:.4f}"
)


print(
    f"Final Test Accuracy : "
    f"{test_accuracy * 100:.2f}%"
)


# ==========================================================
# SAVE MODEL
# ==========================================================

model.save(
    "lstm_model.keras"
)


print(
    "\nlstm_model.keras saved successfully."
)


# ==========================================================
# SAVE TRAINING HISTORY
# ==========================================================

print("\n" + "=" * 60)
print("Saving Training History...")
print("=" * 60)


history_df = pd.DataFrame({

    "Epoch":
        range(
            1,
            EPOCHS + 1
        ),

    "Training Accuracy":
        history.history["accuracy"],

    "Test Accuracy":
        test_callback.test_accuracy,

    "Training Loss":
        history.history["loss"],

    "Test Loss":
        test_callback.test_loss

})


history_df.to_csv(

    "lstm_training_history.csv",

    index=False

)


print(
    "lstm_training_history.csv saved."
)


print(
    "\nTraining History:"
)


print(
    history_df.to_string(
        index=False
    )
)


# ==========================================================
# ACCURACY GRAPH
# ==========================================================

print("\n" + "=" * 60)
print("Generating Accuracy Graph...")
print("=" * 60)


plt.figure(
    figsize=(8, 5)
)


# Training Accuracy
plt.plot(

    range(
        1,
        EPOCHS + 1
    ),

    history.history["accuracy"],

    label="Training Accuracy",

    linewidth=2

)


# Test Accuracy
plt.plot(

    range(
        1,
        EPOCHS + 1
    ),

    test_callback.test_accuracy,

    label="Test Accuracy",

    linewidth=2

)


plt.title(
    "LSTM Model Accuracy"
)


plt.xlabel(
    "Epoch"
)


plt.ylabel(
    "Accuracy"
)


plt.xticks(
    range(
        1,
        EPOCHS + 1
    )
)


plt.legend()


plt.grid(
    True
)


plt.tight_layout()


plt.savefig(

    "lstm_accuracy.png",

    dpi=300,

    bbox_inches="tight"

)


plt.show()


plt.close()


print(
    "lstm_accuracy.png saved successfully."
)


# ==========================================================
# LOSS GRAPH
# ==========================================================

print("\n" + "=" * 60)
print("Generating Loss Graph...")
print("=" * 60)


plt.figure(
    figsize=(8, 5)
)


# Training Loss
plt.plot(

    range(
        1,
        EPOCHS + 1
    ),

    history.history["loss"],

    label="Training Loss",

    linewidth=2

)


# Test Loss
plt.plot(

    range(
        1,
        EPOCHS + 1
    ),

    test_callback.test_loss,

    label="Test Loss",

    linewidth=2

)


plt.title(
    "LSTM Model Loss"
)


plt.xlabel(
    "Epoch"
)


plt.ylabel(
    "Loss"
)


plt.xticks(
    range(
        1,
        EPOCHS + 1
    )
)


plt.legend()


plt.grid(
    True
)


plt.tight_layout()


plt.savefig(

    "lstm_loss.png",

    dpi=300,

    bbox_inches="tight"

)


plt.show()


plt.close()


print(
    "lstm_loss.png saved successfully."
)


# ==========================================================
# GENERATE TEST PREDICTIONS
# ==========================================================

print("\n" + "=" * 60)
print("Generating Test Predictions...")
print("=" * 60)


predictions = model.predict(

    X_test,

    verbose=0

)


predicted_class = np.argmax(

    predictions,

    axis=1

)


actual_class = y_test_integer


# ==========================================================
# CLASSIFICATION REPORT
# ==========================================================

print(
    "\nClassification Report\n"
)


classification_result = classification_report(

    actual_class,

    predicted_class,

    labels=np.arange(
        num_classes
    ),

    target_names=encoder.classes_,

    zero_division=0

)


print(
    classification_result
)


# ==========================================================
# SAVE CLASSIFICATION REPORT
# ==========================================================

with open(
    "lstm_classification_report.txt",
    "w",
    encoding="utf-8"
) as f:

    f.write(
        classification_result
    )


print(
    "lstm_classification_report.txt saved."
)


# ==========================================================
# CONFUSION MATRIX
# ==========================================================

cm = confusion_matrix(

    actual_class,

    predicted_class,

    labels=np.arange(
        num_classes
    )

)


cm_df = pd.DataFrame(

    cm,

    index=encoder.classes_,

    columns=encoder.classes_

)


cm_df.to_csv(
    "lstm_confusion_matrix.csv"
)


print(
    "lstm_confusion_matrix.csv saved."
)


# ==========================================================
# SAVE CONFIGURATION
# ==========================================================

with open(
    "lstm_config.pkl",
    "wb"
) as f:

    pickle.dump(

        {

            "max_length":
                max_length,

            "vocab_size":
                vocab_size,

            "embedding_dim":
                EMBEDDING_DIM,

            "lstm_units":
                LSTM_UNITS,

            "num_classes":
                num_classes,

            "sequence_percentile":
                SEQUENCE_PERCENTILE,

            "test_size":
                TEST_SIZE,

            "epochs":
                EPOCHS,

            "batch_size":
                BATCH_SIZE,

            "dropout_rate":
                DROPOUT_RATE

        },

        f

    )


print(
    "\nlstm_config.pkl saved."
)


# ==========================================================
# FINAL SUMMARY
# ==========================================================

print("\n" + "=" * 60)

print(
    "Training Completed Successfully"
)

print("=" * 60)


print(
    f"\nFinal Test Accuracy: "
    f"{test_accuracy * 100:.2f}%"
)


print(
    f"Final Test Loss: "
    f"{test_loss:.4f}"
)


print("\nGenerated Files:")

print("✓ lstm_model.keras")
print("✓ lstm_tokenizer.pkl")
print("✓ lstm_label_encoder.pkl")
print("✓ lstm_responses.pkl")
print("✓ lstm_config.pkl")
print("✓ lstm_training_history.csv")
print("✓ lstm_classification_report.txt")
print("✓ lstm_confusion_matrix.csv")
print("✓ lstm_accuracy.png")
print("✓ lstm_loss.png")


print(
    "\nYou can now run chatbot.py"
)

print("=" * 60)