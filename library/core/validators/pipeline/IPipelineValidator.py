from abc import ABC, abstractmethod
from library.core.pipeline.Pipeline import Pipeline

class IPipelineValidator(ABC):

    @abstractmethod
    def validate(self, pipeline: Pipeline) -> None:
       pass