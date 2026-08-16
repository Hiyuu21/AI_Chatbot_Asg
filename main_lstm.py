# ==========================================================
# chatbot.py
# LSTM Campus AI Chatbot
# ==========================================================


# ==========================================================
# IMPORT LIBRARIES
# ==========================================================

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import pickle

import numpy as np

from keras.models import load_model

from keras.preprocessing.sequence import (
    pad_sequences
)


# ==========================================================
# IMPORT SHARED PREPROCESSING
# ==========================================================

from preprocess import preprocess_text


# ==========================================================
# CONFIGURATION
# ==========================================================

MODEL_FILE = "model.keras"

TOKENIZER_FILE = "tokenizer.pkl"

ENCODER_FILE = "label_encoder.pkl"

RESPONSES_FILE = "responses.pkl"

CONFIG_FILE = "config.pkl"


# ==========================================================
# CONFIDENCE THRESHOLD
# ==========================================================

CONFIDENCE_THRESHOLD = 0.60


# ==========================================================
# CHECK REQUIRED FILES
# ==========================================================

required_files = [

    MODEL_FILE,

    TOKENIZER_FILE,

    ENCODER_FILE,

    RESPONSES_FILE,

    CONFIG_FILE

]


for file in required_files:

    if not os.path.exists(file):

        raise FileNotFoundError(

            f"Required file not found: {file}"

        )


# ==========================================================
# LOAD MODEL
# ==========================================================

print("=" * 60)

print("Loading Campus AI Chatbot...")

print("=" * 60)


model = load_model(
    MODEL_FILE
)


print(
    "✓ Model Loaded"
)


# ==========================================================
# LOAD TOKENIZER
# ==========================================================

with open(
    TOKENIZER_FILE,
    "rb"
) as f:

    tokenizer = pickle.load(
        f
    )


print(
    "✓ Tokenizer Loaded"
)


# ==========================================================
# LOAD LABEL ENCODER
# ==========================================================

with open(
    ENCODER_FILE,
    "rb"
) as f:

    encoder = pickle.load(
        f
    )


print(
    "✓ Label Encoder Loaded"
)


# ==========================================================
# LOAD RESPONSES
# ==========================================================

with open(
    RESPONSES_FILE,
    "rb"
) as f:

    responses = pickle.load(
        f
    )


print(
    "✓ Responses Loaded"
)


# ==========================================================
# LOAD CONFIGURATION
# ==========================================================

with open(
    CONFIG_FILE,
    "rb"
) as f:

    config = pickle.load(
        f
    )


max_length = config[
    "max_length"
]


print(
    "✓ Configuration Loaded"
)


# ==========================================================
# DISPLAY CONFIGURATION
# ==========================================================

print(
    f"\nMaximum Sequence Length: "
    f"{max_length}"
)


print(
    f"Number of Intents: "
    f"{config['num_classes']}"
)


print(
    f"Vocabulary Size: "
    f"{config['vocab_size']}"
)


# ==========================================================
# PREDICT INTENT
# ==========================================================

def predict_intent(sentence):
    """
    Preprocess the user's question, convert it into
    a sequence, pad it, and use the LSTM model to
    predict the intent.
    """

    # ------------------------------------------------------
    # PREPROCESS USER INPUT
    # ------------------------------------------------------

    processed_sentence = preprocess_text(
        sentence
    )


    # ------------------------------------------------------
    # CHECK EMPTY INPUT
    # ------------------------------------------------------

    if processed_sentence == "":

        return (
            None,
            0.0,
            processed_sentence
        )


    # ------------------------------------------------------
    # CONVERT TEXT TO SEQUENCE
    # ------------------------------------------------------

    sequence = tokenizer.texts_to_sequences(

        [processed_sentence]

    )


    # ------------------------------------------------------
    # CHECK WHETHER TOKENIZER FOUND WORDS
    # ------------------------------------------------------

    if len(sequence[0]) == 0:

        return (

            None,

            0.0,

            processed_sentence

        )


    # ------------------------------------------------------
    # PAD SEQUENCE
    # ------------------------------------------------------

    padded = pad_sequences(

        sequence,

        maxlen=max_length,

        padding="post",

        truncating="post"

    )


    # ------------------------------------------------------
    # MODEL PREDICTION
    # ------------------------------------------------------

    prediction = model.predict(

        padded,

        verbose=0

    )


    # ------------------------------------------------------
    # GET CONFIDENCE
    # ------------------------------------------------------

    confidence = float(

        np.max(
            prediction
        )

    )


    # ------------------------------------------------------
    # GET PREDICTED CLASS
    # ------------------------------------------------------

    predicted_index = int(

        np.argmax(
            prediction
        )

    )


    # ------------------------------------------------------
    # CONVERT CLASS NUMBER TO INTENT NAME
    # ------------------------------------------------------

    intent = encoder.inverse_transform(

        [predicted_index]

    )[0]


    return (

        intent,

        confidence,

        processed_sentence

    )


# ==========================================================
# GET RESPONSE
# ==========================================================

def get_response(intent):
    """
    Retrieve the response associated with
    the predicted intent.
    """

    if intent in responses:

        return responses[intent]


    return (
        "Sorry, I don't have an answer for that."
    )


# ==========================================================
# CHATBOT BANNER
# ==========================================================

print("\n")

print("=" * 60)

print("Campus AI Chatbot")

print("=" * 60)

print(
    "Type your question below."
)

print(
    "Type 'exit', 'quit', or 'bye' to stop."
)

print("=" * 60)


# ==========================================================
# CHAT LOOP
# ==========================================================

while True:

    user_input = input(
        "\nYou : "
    )


    # ------------------------------------------------------
    # EMPTY INPUT
    # ------------------------------------------------------

    if user_input.strip() == "":

        print(
            "Bot : Please enter a question."
        )

        continue


    # ------------------------------------------------------
    # EXIT COMMAND
    # ------------------------------------------------------

    if user_input.lower().strip() in [

        "exit",

        "quit",

        "bye"

    ]:

        print(
            "\nBot : Goodbye! Have a nice day."
        )

        break


    # ------------------------------------------------------
    # PREDICT INTENT
    # ------------------------------------------------------

    intent, confidence, processed_text = (

        predict_intent(
            user_input
        )

    )


    # ------------------------------------------------------
    # EMPTY / INVALID PROCESSED INPUT
    # ------------------------------------------------------

    if intent is None:

        print(
            "\nBot : Sorry, I could not understand "
            "the input."
        )

        continue


    # ------------------------------------------------------
    # DISPLAY DEBUG INFORMATION
    # ------------------------------------------------------

    print(
        f"\nProcessed Question : "
        f"{processed_text}"
    )


    print(
        f"Predicted Intent    : "
        f"{intent}"
    )


    print(
        f"Confidence          : "
        f"{confidence:.2%}"
    )


    # ------------------------------------------------------
    # CONFIDENCE CHECK
    # ------------------------------------------------------

    if confidence < CONFIDENCE_THRESHOLD:

        print(
            "\nBot : Sorry, I don't understand "
            "your question."
        )

        continue


    # ------------------------------------------------------
    # GET RESPONSE
    # ------------------------------------------------------

    response = get_response(
        intent
    )


    # ------------------------------------------------------
    # DISPLAY RESPONSE
    # ------------------------------------------------------

    print(
        "\nBot :",
        response
    )


# ==========================================================
# END
# ==========================================================

print(
    "\nChatbot Closed."
)