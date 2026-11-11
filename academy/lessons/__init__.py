"""Lesson registry.

To add a lesson: create `academy/lessons/<anything>.py` that defines a module-level `LESSON = Lesson(...)`.
It is discovered automatically (files starting with "_" are ignored). Nothing else to edit.
"""
from __future__ import annotations

import importlib
import pkgutil

LOAD_ERRORS: list = []


def all_lessons() -> list:
    LOAD_ERRORS.clear()
    lessons = []
    for mod in pkgutil.iter_modules(__path__):
        if mod.name.startswith("_"):
            continue
        try:
            m = importlib.import_module(f"{__name__}.{mod.name}")
            lessons.append(m.LESSON)
        except Exception as exc:  # a broken lesson must not take the whole app down
            LOAD_ERRORS.append(f"{mod.name}: {exc!r}")
    return sorted(lessons, key=lambda l: (l.level, l.order))


def get_lesson(lesson_id: str):
    return next((l for l in all_lessons() if l.id == lesson_id), None)
