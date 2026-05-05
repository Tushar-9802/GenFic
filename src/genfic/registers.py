"""Single source of truth for the five literary register clusters.

Every other component (ingest, dataset build, cluster validation, training,
inference) reads from `REGISTERS` so renaming a cluster is one edit.

A cluster is:
- `name`: filesystem-safe identifier used in paths and CLI flags
- `display`: human-readable label used in prompts
- `authors`: PD authors on Project Gutenberg whose prose populates the cluster
- `style_hint`: 1–2 sentence description used in the inference system prompt

Cluster validity (separation on the §IV-D metrics) is checked by
`scripts/build_register_clusters.py` — if a cluster's metric bands overlap
others on >2 axes, treat the cluster as invalid and re-curate authors.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Register:
    name: str
    display: str
    authors: tuple[str, ...]
    style_hint: str


REGISTERS: dict[str, Register] = {
    "victorian-formal": Register(
        name="victorian-formal",
        display="Victorian formal realism",
        authors=("Jane Austen", "Anthony Trollope", "George Eliot", "Elizabeth Gaskell"),
        style_hint=(
            "Write in the style of nineteenth-century English realism: measured "
            "periodic sentences, formal diction, free indirect style for character "
            "interiority, careful social observation."
        ),
    ),
    "romantic-ornate": Register(
        name="romantic-ornate",
        display="Romantic ornate",
        authors=("Charlotte Brontë", "Emily Brontë", "Anne Brontë",
                 "Nathaniel Hawthorne", "Walter Scott"),
        style_hint=(
            "Write in the Romantic mode: long sinuous sentences, elevated diction, "
            "extended descriptive passages of landscape and weather, intense first- "
            "or third-person interiority."
        ),
    ),
    "gothic-dark": Register(
        name="gothic-dark",
        display="Gothic dark",
        authors=("Edgar Allan Poe", "Bram Stoker",
                 "Joseph Sheridan Le Fanu", "M. R. James"),
        style_hint=(
            "Write in the Gothic mode: atmospheric and foreboding, archaic diction, "
            "ornate description of ruins and weather, slow accumulation of dread, "
            "the supernatural treated with restraint."
        ),
    ),
    "plain-realist": Register(
        name="plain-realist",
        display="Plain realism",
        authors=("Mark Twain", "Jack London", "Stephen Crane"),
        style_hint=(
            "Write in plain American realism: short concrete sentences, dialogue-"
            "forward scenes, vernacular speech where appropriate, no ornamental "
            "diction, physical action over interiority."
        ),
    ),
    "modernist-spare": Register(
        name="modernist-spare",
        display="Modernist spare",
        authors=("James Joyce", "Sherwood Anderson", "Ring Lardner"),
        style_hint=(
            "Write in the spare modernist mode: declarative sentences, paratactic "
            "rhythm, ellipsis and implication over explanation, ordinary objects "
            "carrying weight, restrained dialogue."
        ),
    ),
}


def get(name: str) -> Register:
    if name not in REGISTERS:
        raise KeyError(f"Unknown register {name!r}. Known: {sorted(REGISTERS)}")
    return REGISTERS[name]


def names() -> list[str]:
    return list(REGISTERS)
