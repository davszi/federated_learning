from abc import ABC, abstractmethod

class Logger(ABC):
    @abstractmethod
    def log(self, name: str, data: dict):
        pass
