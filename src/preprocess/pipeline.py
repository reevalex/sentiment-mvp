import joblib
import pandas as pd
from clean_text import clean_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# ---------------------------------------------------
# Step 1

class SentimentPipeline:
  def __init__(self, max_features=5000, test_size=0.2, random_state=123):
    self.max_features = max_features
    self.test_size = test_size
    self.random_state =random_state
    self.vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2)) # unigram + bigram
    self.is_fitted = False

  def preprocess(self, df):
    df = df.copy()
    df["text_clean"] = df.text.apply(clean_text)
    df = df[df.text_clean.str.len() > 0]
    return df
  
  def vectorize(self, texts):
    X = self.vectorizer.fit_transform(texts)
    feature_names = self.vectorizer.get_feature_names_out()
    X_df = pd.DataFrame(X.toarray(), columns=feature_names)
    self.is_fitted = True
    return X_df, self.vectorizer

  def split_data(self, X, y):
    return train_test_split(X, y, test_size=self.test_size, random_state=self.random_state, stratify=y)
  
  def run_full(self, df, text_col="text", label_col="label"):
    df_clean = self.preprocess(df)
    X_temp, self.vectorizer = self.vectorize(df_clean[text_col])
    
    X_train, X_test, y_train, y_test = self.split_data(X_temp, df_clean[label_col])
    
    print(f"Processed: {len(X_train)} train, {len(X_test)} test samples")
    print(f"Features: {X_train.shape[1]}")
    print(f"Labels: {y_train.shape}")
    
    return (X_train, X_test, y_train, y_test, df_clean)
  
if __name__ == "__main__":
  df_train = pd.read_csv("data/raw/imdb_train.csv")
  pipe = SentimentPipeline(max_features=5000)
  X_train, X_test, y_train, y_test, df_clean = pipe.run_full(df_train)
  
  joblib.dump(pipe.vectorizer, "data/models/vectorizer.pkl")
  X_train.to_csv("data/proceseed/train_features.csv", index=False)
  X_test.to_csv("data/proceseed/test_features.csv", index=False)
  y_train.to_csv("data/proceseed/train_labels.csv", index=False)
  y_test.to_csv("data/proceseed/test_labels.csv", index=False)
  df_clean.to_csv("data/proceseed/train_clean.csv", index=False)
  
  print("Processed data saved!")
