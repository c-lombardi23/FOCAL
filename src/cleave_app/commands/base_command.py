"""Abstract class to define all main function logic."""

import traceback
from abc import ABC, abstractmethod


class BaseCommand(ABC):
    """An abstract class for all the command classes to inherit from"""

    def execute(self, config) -> None:
        """Runs the command"""
        try:
            self._execute_command(config)
        except Exception as e:
            print(f"Error during command: {self.__class__.__name__}: {e}")
            traceback.print_exc()
            raise

    @abstractmethod
    def _execute_command(self, config) -> None:
        pass
