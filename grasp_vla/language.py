"""Instruction -> plan.

The default planner is a small rule-based parser, so every result in this repo
is reproducible offline. `llm_plan` is an optional drop-in that asks an LLM for
the same JSON plan; it is only used when an API key is available and is not
part of the reported evaluation.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass

COLOR_WORDS = {
    "red": "red", "crimson": "red", "scarlet": "red",
    "green": "green", "lime": "green",
    "blue": "blue", "navy": "blue",
    "yellow": "yellow", "golden": "yellow",
    "purple": "purple", "violet": "purple",
    "orange": "orange",
}
SHAPE_WORDS = {
    "cube": "cube", "block": "cube", "box": "cube", "square": "cube",
    "cylinder": "cylinder", "can": "cylinder", "tube": "cylinder", "puck": "cylinder",
    "ball": "ball", "sphere": "ball", "marble": "ball",
}
PLACE_WORDS = ("tray", "bin", "basket", "container")


@dataclass
class Step:
    action: str          # "pick_place" (pick, then drop in the tray) or "pick" (pick and hold)
    target: str          # free-text phrase, e.g. "red cube"
    color: str | None    # canonical attribute if the parser recognised one
    shape: str | None

    def to_dict(self):
        return {"action": self.action, "target": self.target, "color": self.color, "shape": self.shape}


def _parse_phrase(words: list[str]):
    color = next((COLOR_WORDS[w] for w in words if w in COLOR_WORDS), None)
    shape = next((SHAPE_WORDS[w] for w in words if w in SHAPE_WORDS), None)
    return color, shape


def parse(instruction: str) -> list[Step]:
    """Rule-based planner. Handles one or more targets joined by 'and' / 'then'."""
    text = instruction.lower().strip().rstrip(".")
    place = any(w in text for w in PLACE_WORDS)
    # drop the destination clause, then split into object phrases
    text = re.sub(r"\b(in|into|to|onto|inside)\s+the\s+(" + "|".join(PLACE_WORDS) + r")\b", " ", text)
    chunks = re.split(r"\b(?:and then|then|and|,)\b", text)
    steps = []
    for chunk in chunks:
        words = re.findall(r"[a-z]+", chunk)
        color, shape = _parse_phrase(words)
        if color is None and shape is None:
            continue
        phrase = " ".join(x for x in (color, shape) if x) or chunk.strip()
        steps.append(Step("pick_place" if place else "pick", phrase, color, shape))
    return steps


PLAN_PROMPT = """You control a robot arm above a table. Turn the instruction into a JSON list of
steps. Each step is {{"action": "pick_place" | "pick", "target": "<colour> <shape>",
"color": one of {colors} or null, "shape": one of ["cube","cylinder","ball"] or null}}.
Use "pick_place" when the object should end up in the tray. Reply with JSON only.
Instruction: {instruction}"""


def llm_plan(instruction: str, model: str = "claude-sonnet-5-5") -> list[Step]:
    """Optional LLM planner (needs `pip install anthropic` and ANTHROPIC_API_KEY)."""
    import anthropic  # imported lazily so the package works without it

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY is not set; use parse() instead")
    colors = sorted(set(COLOR_WORDS.values()))
    msg = anthropic.Anthropic().messages.create(
        model=model, max_tokens=400,
        messages=[{"role": "user", "content": PLAN_PROMPT.format(colors=colors, instruction=instruction)}])
    data = json.loads(msg.content[0].text)
    return [Step(d["action"], d["target"], d.get("color"), d.get("shape")) for d in data]
