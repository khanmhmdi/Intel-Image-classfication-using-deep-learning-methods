from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Optional

class model_handler(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def set_next(self,model:model_handler) -> model_handler:
        pass
    @abstractmethod
    def handle(self,request) -> Optional[str]:
        pass

    