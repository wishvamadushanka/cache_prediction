import json


def load_settings(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)
