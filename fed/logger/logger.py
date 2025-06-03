from abc import ABC, abstractmethod

class Logger(ABC):
    @abstractmethod
    def log(self, data: dict, name: str = '', step: int | float = None):
        pass
