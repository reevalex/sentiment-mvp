from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud
from pprint import pprint

plt.style.use("default")

# ---------------------------------------------------
# EDA

train_df = pd.read_csv("../data/raw/imdb_train.csv")
test_df = pd.read_csv("../data/raw/imdb_test.csv")
df = pd.concat([train_df, test_df], keys=["train", "test"]).reset_index(level=0).rename(columns={"level_0": "split"})

df.shape
df.columns.to_list()
df.head(3)

df.label.value_counts(normalize=True) # 0.5/0.5
df.isnull().sum()

df["length"] = df.text.str.len()
print(df.groupby("split")["length"].describe().T)

nltk.download("stopwords", quiet=True)
stop_words = set(stopwords.words("english"))

def get_top_words(text_series, n=10):
  words = " ".join(text_series).lower().split()
  filtered = [w for w in words if w.isalpha() and w not in stop_words]
  return Counter(filtered).most_common(n)

pos_train = df[(df.split == "train") & (df.label == 1)].text
pprint(get_top_words(pos_train))

neg_train = df[(df.split == "train") & (df.label == 0)].text
pprint(get_top_words(neg_train))

# ---------------------------------------------------
# Visualization

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

sns.countplot(data=df, x="label", hue="split", ax=axes[0, 0])
axes[0, 0].set_title("Label Distribution")

df.length.hist(bins=50, ax=axes[0, 1])
axes[0, 1].set_title("Review Lengths")
axes[0, 1].set_yscale("log")

sns.boxplot(data=df, x="label", y="length", ax=axes[1, 0])
axes[1, 0].set_title("Length by Sentiment")

wc_pos = WordCloud().generate(" ".join(pos_train))
axes[1, 1].imshow(wc_pos)
axes[1, 1].axis("off")

plt.tight_layout()
plt.show()


