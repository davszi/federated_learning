from pathlib import Path

from .logger import Logger

class FileLogger(Logger):
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def log(self, name: str, data: dict):
        file_path = self.base_path / f"{name}.csv"
        file_exists = file_path.exists()

        with file_path.open(mode="a") as file:
            if not file_exists:
                file.write(",".join(data.keys()) + "\n")
            file.write(",".join(str(value) for value in data.values()) + "\n")
