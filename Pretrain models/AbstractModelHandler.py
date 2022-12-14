import models
from abc import ABC, abstractmethod
from typing import Any, Optional

class AbstractModelHandler(models):
    
    _next_handler : models = None
    
    def set_next_model(self, handler:models) -> models :
        self._next_handler = handler

    @abstractmethod
    def handle(self, request:Any ):
        if self._next_handler:
            return self._next_handler.handle(request)

        return None    
