import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
 

stop_words =set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text, remove_stopwords=True):
  if pd.isna(text) or not isinstance(text, str):
    return ""
  
  text = text.lower()
  text = re.sub(r"<.*?>", "", text) # strip HTML
  text = re.sub(r"[^a-zA-Z\s]", "", text) # remove punctuation/numbers
  
  tokens = word_tokenize(text)
  
  if remove_stopwords:
    tokens = [word for word in tokens if word not in stop_words]
  
  tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalpha()]
  
  return " ".join(tokens)

if __name__ == "__main__":
  sample = "This movie was <b>amazing</b>!!! I loved it, running through the fields."
  print(f"Raw: {sample}")
  print(f"Cleaned: {clean_text(sample)}")