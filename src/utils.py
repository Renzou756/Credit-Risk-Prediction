import json
from typing import Dict

def sigmoid(x: float) -> float:
    return 1 / (1 + np.exp(-x))

def load_json(path: str) -> Dict:
    with open(path, 'r') as f:
        return json.load(f)

def save_json(data: Dict, path: str):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)