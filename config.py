import json

def load_config():
    """
    Load configuration from a JSON file (e.g., for model paths, API keys, etc.)
    """
    with open("config.json") as f:
        config = json.load(f)
    return config
