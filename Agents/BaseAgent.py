from abc import ABC, abstractmethod
class BaseAgent(ABC):
    def __init__(self, **config):
         for k, v in config.items():
            setattr(self, k, v)

    @abstractmethod
    def choose_action(self, state):
        pass

    @abstractmethod
    def store_in_memory(self, state, action, reward, next_state, done):
        pass

    @abstractmethod
    def learn(self):
        pass
