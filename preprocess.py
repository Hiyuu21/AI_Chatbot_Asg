import re
import os
import random
import pandas as pd
from collections import defaultdict

try:
    from nltk.stem import WordNetLemmatizer
    from nltk.corpus import wordnet
    import nltk

    for resource in ["wordnet", "omw-1.4", "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng"]:
        try:
            nltk.data.find(f"corpora/{resource}")
        except LookupError:
            try:
                nltk.data.find(f"taggers/{resource}")
            except LookupError:
                nltk.download(resource, quiet=True)

    _lemmatizer = WordNetLemmatizer()
    _LEMMATIZER_AVAILABLE = True
except ImportError:
    _LEMMATIZER_AVAILABLE = False
    print("-- WARNING: nltk not installed. Lemmatization disabled.")

def _get_wordnet_pos(nltk_tag):
    if nltk_tag.startswith('V'): return wordnet.VERB
    elif nltk_tag.startswith('J'): return wordnet.ADJ
    elif nltk_tag.startswith('R'): return wordnet.ADV
    else: return wordnet.NOUN

def _lemmatize(text):
    if not _LEMMATIZER_AVAILABLE: return text
    words = text.split()
    if not words: return text
    tagged = nltk.pos_tag(words)
    lemmatized = [_lemmatizer.lemmatize(word, _get_wordnet_pos(tag)) for word, tag in tagged]
    return " ".join(lemmatized)

def preprocess_text(text):
    text = text.lower()
    contractions = {
        "can't": "cannot", "won't": "will not", "don't": "do not", "doesn't": "does not",
        "didn't": "did not", "isn't": "is not", "aren't": "are not", "wasn't": "was not",
        "weren't": "were not", "couldn't": "could not", "wouldn't": "would not",
        "shouldn't": "should not", "i'm": "i am", "you're": "you are", "we're": "we are",
        "they're": "they are", "he's": "he is", "she's": "she is", "it's": "it is",
        "i've": "i have", "you've": "you have", "we've": "we have", "they've": "they have",
        "i'll": "i will", "you'll": "you will", "we'll": "we will", "they'll": "they will",
        "what's": "what is", "where's": "where is", "when's": "when is", "who's": "who is",
        "how's": "how is"
    }
    for k, v in contractions.items():
        text = text.replace(k, v)
        
    replacements = {
        r'\blib\b': 'library', r'\bfoundatn\b': 'foundation', r'\bsem\b': 'semester',
        r'\bpwd\b': 'password', r'\bwat\b': 'what', r'\byr\b': 'year',
        r'\bfafb\b': 'accountancy finance business faculty', r'\bfoas\b': 'applied sciences faculty',
        r'\bfobe\b': 'built environment faculty', r'\bfcci\b': 'communication creative industries faculty',
        r'\bfocs\b': 'computing information technology faculty', r'\bfoet\b': 'engineering technology faculty',
        r'\bfssh\b': 'social science humanities faculty'
    }
    for k, v in replacements.items():
        text = re.sub(k, v, text)

    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return _lemmatize(text)


"""
Loading dataset from excel

This loads the data and includes the 'SeedGroup' column. 'SeedGroup' links 
together questions that are just slightly worded differently. We need this 
so similar questions don't end up in both the training and testing sets, 
which would let the model "cheat" and artificially boost its test score.
"""


def load_and_clean_data(file_path):
    """Loads dataset, validates columns, drops empty/duplicate rows, and applies text processing."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    print(f"\n-- Loading and cleaning data from {file_path}.")
    df = pd.read_excel(file_path)

    required_columns = ["Question", "Intent", "Response"]
    for col in required_columns:
        if col not in df.columns:
            raise Exception(f"Missing column: {col}")

    df = df.dropna(subset=required_columns)
    df["Question"] = df["Question"].astype(str)
    
    before_duplicates = len(df)
    df = df.drop_duplicates(subset=["Question"], keep="first")
    print(f"-- Duplicate Questions Removed: {before_duplicates - len(df)}")

    # Apply preprocessing and strip out rows that resulted in empty strings
    df["Processed_Question"] = df["Question"].apply(preprocess_text)
    df = df[df["Processed_Question"].str.strip() != ""]
    df = df.reset_index(drop=True)

    has_group_col = 'SeedGroup' in df.columns
    if not has_group_col:
        print("-- WARNING: 'SeedGroup' column not found. Falling back to row-level grouping.")
    has_coarse_col = 'CoarseIntent' in df.columns

    raw_data = []
    intent_responses = {}

    for index, row in df.iterrows():
        q_processed = str(row['Processed_Question'])
        fine_intent = str(row['Intent']).strip()
        target_intent = str(row['CoarseIntent']).strip() if has_coarse_col else fine_intent
        response = str(row['Response']).strip()
        group_str = str(row['SeedGroup']).strip() if has_group_col else f"{target_intent}_row{index}"

        # Tuple structure: (Processed Question, Classification Intent, Group ID)
        raw_data.append((q_processed, target_intent, group_str))

        if fine_intent not in intent_responses:
            intent_responses[fine_intent] = response

    print(f"-- Successfully cleaned and loaded {len(raw_data)} questions.")
    return raw_data, intent_responses



"""
Group-aware train/test split

A standard random split just shuffles rows. That allows nearly identical 
questions to land in both training and testing, giving us a fake high score.

This function splits the data by 'SeedGroup' instead. If a group of similar 
questions is assigned to training, ALL of them go to training. This keeps 
the test honest while keeping the categories balanced.

NOTE: If a category only has ONE group of questions, splitting by group 
would send 100% of it to the test set, leaving the model nothing to learn from. 
When that happens, we fall back to a standard random split just for that category.
"""


def group_aware_split(raw_data, test_size=0.2, seed=42):
    """Splits data by SeedGroup to prevent data leakage between train and test sets."""
    rng = random.Random(seed)
    by_intent_groups = defaultdict(lambda: defaultdict(list))
    
    for row in raw_data:
        intent = row[1]
        group = row[2]
        by_intent_groups[intent][group].append(row)

    train_data, test_data = [], []
    fallback_intents = []

    print("\n-- Performing group-aware split (per intent):")
    for intent, groups in by_intent_groups.items():
        group_ids = list(groups.keys())
        rng.shuffle(group_ids)
        total_rows = sum(len(groups[g]) for g in group_ids)
        target_test_rows = round(total_rows * test_size)

        if len(group_ids) < 2:
            fallback_intents.append(intent)
            all_rows = groups[group_ids[0]]
            rng.shuffle(all_rows)
            test_rows = all_rows[:target_test_rows]
            train_rows = all_rows[target_test_rows:]
            test_data.extend(test_rows)
            train_data.extend(train_rows)
        else:
            test_group_ids = set()
            test_row_count = 0
            for g in group_ids:
                if test_row_count >= target_test_rows: break
                test_group_ids.add(g)
                test_row_count += len(groups[g])

            for g in group_ids:
                if g in test_group_ids: test_data.extend(groups[g])
                else: train_data.extend(groups[g])

    rng.shuffle(train_data)
    rng.shuffle(test_data)
    print(f"-- Split complete. Train: {len(train_data)} rows | Test: {len(test_data)} rows\n")
    return train_data, test_data, set(fallback_intents)