"""Load system prompts from the project YAML, independently of the working directory."""
from functools import cache
from pathlib import Path

import yaml

PROMPTS_PATH = Path(__file__).resolve().parent.parent / "configs" / "prompts.yaml"


@cache
def load_prompt(name: str) -> str:
    """Read a named prompt once per process; restart after editing the YAML."""
    with PROMPTS_PATH.open(encoding="utf-8") as file:
        prompts = yaml.safe_load(file)
    if not isinstance(prompts, dict):
        raise ValueError(f"Expected a mapping in {PROMPTS_PATH}")
    prompt = prompts.get(name)
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(f"Missing or empty prompt '{name}' in {PROMPTS_PATH}")
    return prompt
