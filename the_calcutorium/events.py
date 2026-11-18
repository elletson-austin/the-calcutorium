"""A tiny observer-pattern helper, so the core doesn't need Qt to announce changes.

    changed = Event()
    changed.subscribe(lambda obj: print("changed", obj))
    changed.emit(some_obj)
"""

from typing import Callable


class Event:
    def __init__(self) -> None:
        self._listeners: list[Callable[..., None]] = []

    def subscribe(self, listener: Callable[..., None]) -> None:
        self._listeners.append(listener)

    def emit(self, *args) -> None:
        for listener in list(self._listeners):
            listener(*args)
