from typing import Literal

from .logger import Logger
from .file_logger import FileLogger
from .wandb_logger import WandbLogger


class LoggerFactory:
    @classmethod
    def create(cls, name: str, run_name: str, config: dict = None) -> Logger:
        if name == 'file':
            return FileLogger(base_path=f"files/logs/{run_name}", config=config)
        elif name == 'wandb':
            return WandbLogger(run_name=run_name, config=config)
        else:
            raise ValueError(f'Unknown logger name: {name}')