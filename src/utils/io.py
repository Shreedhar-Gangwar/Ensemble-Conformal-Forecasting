import yaml
from typing import Dict

def load_config(path: str) -> Dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)
