"""Plain dataclasses that describe content. Lessons, projects, exercises and concepts are just data."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass(frozen=True)
class Concept:
    num: int                        # 1..36, the tracked concept number
    id: str
    name: str
    level: int
    summary: str                    # one plain-English line


@dataclass
class Question:
    id: str
    kind: str                       # "mc" | "short" | "numeric"
    prompt: str
    explanation: str                # why the right answer is right
    category: str = "Conceptual"    # "Conceptual" | "Application"
    concept: str = ""               # concept id; empty = the lesson's first concept
    options: list = field(default_factory=list)   # mc only
    notes: list = field(default_factory=list)     # mc only: one-line reason per option
    answer: object = None           # mc: index of correct option; numeric: float
    tolerance: float = 0.0          # numeric only (absolute)
    keywords: list = field(default_factory=list)  # short: list of synonym groups
    model_answer: str = ""          # short: shown after grading


@dataclass
class MathBlock:
    title: str
    latex: str
    where: str                      # markdown bullets: every symbol explained
    note: str = ""


@dataclass
class Lesson:
    """The 10-step teaching sequence. Field order mirrors the order on the page."""
    id: str
    level: int
    order: int
    title: str
    subtitle: str
    minutes: int
    concept_ids: list               # concepts this lesson teaches (first = primary)
    simple: str                     # 1. very simple explanation
    problem: str                    # 2. what problem it solves
    why: str
    when_useful: str
    when_not: str
    compare: str                    #    comparison with related methods
    sports: str                     # 3. college football example
    takeaway: str
    visual: Optional[Callable[[], None]]   # 4. visual explanation
    visual_caption: str
    math: list                      # 5-6. math, introduced gradually, every symbol explained
    code: str                       # 7. simple Python example
    code_notes: str
    walkthrough: list               # 8. [(code fragment, plain-English explanation), ...]
    psfal: str                      # 9. PSFAL application
    quiz: list                      # 10. knowledge check

    @property
    def code_label(self) -> str:
        return f"L{self.level}.{self.order}"

    def question_concept(self, q: Question) -> str:
        return q.concept or self.concept_ids[0]


@dataclass
class Exercise:
    id: str
    level: str                      # Beginner | Intermediate | Advanced
    title: str
    goal: str
    concepts: list                  # display tags
    starter: str
    solution: str
    hint: str
    concept_ids: list = field(default_factory=list)


@dataclass
class Track:
    """One half of a project: a Learning Lab or a Production Candidate."""
    summary: str
    steps: list
    code: str = ""                  # runnable starter (Learning Lab)
    limitations: list = field(default_factory=list)
    baseline: str = ""              # Production Candidate fields
    metric: str = ""
    validation: str = ""
    gates: list = field(default_factory=list)


@dataclass
class Project:
    id: str
    level: int
    slot: str                       # "", "A" or "B"
    title: str
    objective: str
    concept_ids: list
    dataset_id: str                 # key into academy.datasets.SPECS
    psfal: str
    learning: Track
    production: Track
    questions: list
    interactive: Optional[Callable[[], None]] = None   # full custom page (Level 1)
    status: str = "starter"         # "built" | "starter"

    @property
    def label(self) -> str:
        return f"P{self.level}{self.slot}"
