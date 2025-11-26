import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Ensure NLTK data is downloaded (safe to call multiple times)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

def clean_text(doc):
    """
    Cleans the input text by:
    1. Removing special characters and digits.
    2. Converting to lowercase.
    3. Tokenizing.
    4. Removing stop words.
    5. Lemmatizing.
    """
    if not isinstance(doc, str):
        return ""
        
    # let's define a regex to match special characters and digits
    regex = '[^a-zA-Z.]'
    doc = re.sub(regex, ' ', doc)
    # convert to lowercase
    doc = doc.lower()
    # tokenization
    tokens = nltk.word_tokenize(doc)
    # Stop word removal 
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
    # join and return 
    return ' '.join(lemmatized_tokens)
