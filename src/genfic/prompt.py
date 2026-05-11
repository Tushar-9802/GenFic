"""Canonical prompt format shared by dataset build, training, and inference.

There is exactly one supported prompt format for the register adapters:

    [INST] Continue this scene in {register.display} style: {brief} [/INST] {response}</s>

`build_dataset.py` constructs the training rows in this shape; `inference/generate.py`
must reproduce the prompt prefix character-for-character so the adapter is queried
on-distribution. Any drift (extra system message, different verb, missing trailing
space) shifts the model off the surface it was trained on.

The trailing space after `[/INST]` is intentional — it's part of the training
prefix the model has seen, so it must appear at inference too. Do not "tidy" it.
"""

from __future__ import annotations

from genfic.registers import REGISTERS, Register


def _resolve(register: str | Register) -> Register:
    if isinstance(register, Register):
        return register
    if register not in REGISTERS:
        raise KeyError(f"Unknown register {register!r}. Known: {sorted(REGISTERS)}")
    return REGISTERS[register]


def format_prompt(register: str | Register, brief: str) -> str:
    """Build the training-equivalent prompt prefix (everything before the response).

    The returned string ends with a single trailing space — the chunk/response
    is appended directly, no separator.
    """
    reg = _resolve(register)
    return f"[INST] Continue this scene in {reg.display} style: {brief} [/INST] "
