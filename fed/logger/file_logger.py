from pathlib import Path
import yaml

from .logger import Logger

class FileLogger(Logger):
    def __init__(self, base_path: str, config: dict | None = None):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.max_indices = {}

        if config is not None:
            config_path = self.base_path / "config.yaml"
            with open(config_path, "w") as f:
                yaml.dump(config, f)


    def log(self, data: dict, name: str = '', step: int | float = None):

        if step is None:
            step = self.max_indices.get(name, 0) + 1
        self.max_indices[name] = max(step, self.max_indices.get(name, 0))

        for key, value in data.items():
            file_name = f"{name}_{key}.csv" if name != '' else f'{key}.csv'
            file_path = self.base_path / file_name
            with open(file_path, "a", newline="") as file:
                file.write(f"{step},{value}\n")
