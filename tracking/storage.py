import json
import os
from datetime import datetime


class Storage:
    def __init__(self, path="automl/experiments/logs.json"):
        self.path = path
        os.makedirs(os.path.dirname(self.path), exist_ok=True)

        if not os.path.exists(self.path):
            with open(self.path, "w") as f:
                json.dump([], f)

    def save(self, record):
        with open(self.path, "r") as f:
            data = json.load(f)

        data.append(record)

        with open(self.path, "w") as f:
            json.dump(data, f, indent=4)

    def generate_run_id(self):
        return datetime.now().strftime("%Y%m%d%H%M%S")