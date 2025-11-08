import urllib.request
import tarfile
import os
import glob
import pandas as pd
from pathlib import Path

# ---------------------------------------------------
# Step 1

data_dir = Path("data/raw")
data_dir.mkdir(parents=True, exist_ok=True)
tar_path = data_dir / "aclImdb_v1.tar.gz"

if not tar_path.exists():
  url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
  print(f"Downloading {url} ... (grab coffee, ~80MB)")
  urllib.request.urlretrieve(url, tar_path)
  print("Download Complete!")
else:
  print("Tar.gz already exists-skipping download.")

# ---------------------------------------------------
# Step 2
  
extract_dir = data_dir / "aclImdb"
if not extract_dir.exists():
  print("Extracting...")
  with tarfile.open(tar_path, "r:gz") as tar:
    tar.extractall(data_dir)
    print("Extraction complete!")
else:
  print("Extracted folder already exists-skipping.")
  
# ---------------------------------------------------
# Step 3

def load_imdb_folder(folder_path):
  data = []
  folder_path = Path(folder_path)
  for label_dir in ["pos", "neg"]:
    label = 1 if label_dir == "pos" else 0
    dir_path = folder_path / label_dir
    if not dir_path.exists():
      continue
    for txt_file in glob.glob(str(dir_path / "*txt")):
      with open(txt_file, "r", encoding="utf-8") as f:
        text = f.read().strip()
        if text:
          data.append({"text": text, "label": label})
  return pd.DataFrame(data)

train_dir = extract_dir / "train"
print("Parsing train data...")
train_df = load_imdb_folder(train_dir)
print(f"Train: {len(train_df)} samples")

test_dir = extract_dir / "test"
print("Parsing test data...")
test_df = load_imdb_folder(test_dir)
print(f"Test: {len(test_df)} samples")

train_df.to_csv(data_dir / "imdb_train.csv", index=False)
test_df.to_csv(data_dir / "imdb_test.csv", index=False)
print(f"Saved CSVs to data/raw/! Total samples: {len(train_df) + len(test_df)}")