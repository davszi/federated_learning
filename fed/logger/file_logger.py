from pathlib import Path

from .logger import Logger

class FileLogger(Logger):
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.max_indices = {}

    def log(self, data: dict, name: str = '', step: int | float = None):
        if step is None:
            step = self.max_indices.get(name, 0) + 1
        self.max_indices[name] = max(step, self.max_indices.get(name, 0))

        for key, value in data.items():
            file_path = self.base_path / f"{name}_{key}.csv"
            with open(file_path, "a", newline="") as file:
                file.write(f"{step},{value}\n")
