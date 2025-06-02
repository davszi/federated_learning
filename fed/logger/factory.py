from typing import Literal

from .logger import Logger
from .file_logger import FileLogger
from .wandb_logger import WandbLogger


class LoggerFactory:
    @classmethod
    def create(cls, name: Literal['file', 'wandb'], **kwargs) -> Logger:
        if name == 'file':
            return FileLogger(base_path=f"logs/{kwargs.get('run_name')}")
        elif name == 'wandb':
            return WandbLogger(run_name=kwargs.get('run_name'), config=kwargs.get('config'))
        else:
            raise ValueError(f'Unknown logger name: {name}')

