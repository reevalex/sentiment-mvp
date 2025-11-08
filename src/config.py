import yaml
from pathlib import Path

def load_config(config_path="config.yaml"):
  with open(config_path, "r") as f:
    config = yaml.safe_load(f)
    
  project_root = Path(config["project_root"]).resolve()
  for section in ["data", "models"]:
    if section in config:
      for key, val in config[section].items():
        if isinstance(val, str) and not val.startswith("/"): 
          config[section][key] = str(project_root / val)
  
  return config


if __name__ == "__main__":
  cfg = load_config()
  print(f"Raw train: {cfg["data"]["raw_train"]}")
  print(f"Max features: {cfg["pipeline"]["max_features"]}")