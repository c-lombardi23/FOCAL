from abc import ABC, abstractmethod

class BaseModelPipeline(ABC):

    @abstractmethod
    def train(self, *args, **kwargs):
        """Train model"""
        pass

    @abstractmethod
    def predict(self, *args, **kwargs):
        """Make predictions for model"""
        pass

    @abstractmethod
    def save(self, *args, **kwargs):
        """Save the model"""
        pass

    @abstractmethod
    def load(self, *args, **kwargs):
        """Load the model from path"""
        pass
