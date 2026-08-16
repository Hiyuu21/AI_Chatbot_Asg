import re

try:
    from nltk.stem import WordNetLemmatizer
    from nltk.corpus import wordnet
    import nltk

    for resource in ["wordnet", "omw-1.4", "averaged_perceptron_tagger",
                      "averaged_perceptron_tagger_eng"]:
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
    print("-- WARNING: nltk not installed. Run 'pip install nltk' to enable "
          "lemmatization (e.g. 'lost'/'losing' -> 'lose'). Falling back to "
          "no lemmatization for now.")


def _get_wordnet_pos(nltk_tag):
    """Map NLTK's POS tag to the tag WordNetLemmatizer expects, so verbs
    like 'lost' correctly lemmatize to 'lose' instead of being left as a
    noun/unchanged. Defaults to noun if the tag isn't recognized."""
    if nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


def _lemmatize(text):
    if not _LEMMATIZER_AVAILABLE:
        return text
    words = text.split()
    if not words:
        return text
    tagged = nltk.pos_tag(words)
    lemmatized = [_lemmatizer.lemmatize(word, _get_wordnet_pos(tag)) for word, tag in tagged]
    return " ".join(lemmatized)


def preprocess_text(text):
    """
    Clean the raw input to convert them into lower caps, and remove the contractions, extra whitespaces, and punctuations
    """
    # Convert the 'text' variable to lowercase
    text = text.lower()

    # Replace contractions
    text = text.replace("can't", "cannot")
    text = text.replace("won't", "will not")

    text = text.replace("don't", "do not")
    text = text.replace("doesn't", "does not")
    text = text.replace("didn't", "did not")

    text = text.replace("isn't", "is not")
    text = text.replace("aren't", "are not")

    text = text.replace("wasn't", "was not")
    text = text.replace("weren't", "were not")

    text = text.replace("couldn't", "could not")
    text = text.replace("wouldn't", "would not")
    text = text.replace("shouldn't", "should not")

    text = text.replace("i'm", "i am")
    text = text.replace("you're", "you are")
    text = text.replace("we're", "we are")
    text = text.replace("they're", "they are")

    text = text.replace("he's", "he is")
    text = text.replace("she's", "she is")
    text = text.replace("it's", "it is")

    text = text.replace("i've", "i have")
    text = text.replace("you've", "you have")
    text = text.replace("we've", "we have")
    text = text.replace("they've", "they have")

    text = text.replace("i'll", "i will")
    text = text.replace("you'll", "you will")
    text = text.replace("we'll", "we will")
    text = text.replace("they'll", "they will")

    text = text.replace("what's", "what is")
    text = text.replace("where's", "where is")
    text = text.replace("when's", "when is")
    text = text.replace("who's", "who is")
    text = text.replace("how's", "how is")
    
    text = re.sub(r'\blib\b', 'library', text)
    text = re.sub(r'\bfoundatn\b', 'foundation', text)
    text = re.sub(r'\bsem\b', 'semester', text)
    text = re.sub(r'\bpwd\b', 'password', text)
    text = re.sub(r'\bwat\b', 'what', text)
    text = re.sub(r'\byr\b', 'year', text)
    text = re.sub(r'\bfafb\b', 'accountancy finance business faculty', text)
    text = re.sub(r'\bfoas\b', 'applied sciences faculty', text)
    text = re.sub(r'\bfobe\b', 'built environment faculty', text)
    text = re.sub(r'\bfcci\b', 'communication creative industries faculty', text)
    text = re.sub(r'\bfocs\b', 'computing information technology faculty', text)
    text = re.sub(r'\bfoet\b', 'engineering technology faculty', text)
    text = re.sub(r'\bfssh\b', 'social science humanities faculty', text)

    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)

    # Remove extra spaces
    # re.sub(r'\s+', ' ', text) collapses multiple internal spaces into one, \s means whitespace, + means one or more
    # .strip() cleans up the leading and trailing edges
    text = re.sub(r'\s+', ' ', text).strip()

    # Lemmatize each word to its base form (e.g. "lost"/"losing"/"loses" -> "lose")
    # so word-form variants that never co-occurred in training still map to
    # the same token the model learned to associate with the correct intent.
    # POS-tagged so verbs, adjectives, and adverbs lemmatize correctly, not
    # just nouns (WordNetLemmatizer defaults to noun otherwise).
    text = _lemmatize(text)

    return text