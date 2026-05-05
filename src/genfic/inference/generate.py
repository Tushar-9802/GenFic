"""Generation pipeline: load Mistral-Instruct base + a register adapter, run a brief.

The system prompt is register-aware: pass `register="victorian-formal"` and the
generator reads `src/genfic/registers.py` to inject the appropriate style hint.
Pass `system=...` to override entirely.

Examples
--------
    from genfic.inference.generate import GenFicGenerator

    gen = GenFicGenerator(
        adapter_path="runs/victorian-formal/checkpoint-500",
        register="victorian-formal",
    )
    text = gen.generate(
        brief="An afternoon visit to the parsonage; the vicar's wife is unwell.",
        max_new_tokens=1500,
        temperature=0.85,
    )
    print(text)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from genfic.registers import REGISTERS

DEFAULT_BASE = "mistralai/Mistral-7B-Instruct-v0.2"

GENERIC_SYSTEM = (
    "You are an expert prose writer. Continue the requested scene in literary "
    "English, no preamble or meta-commentary, no headings."
)


def _system_for_register(register: str | None) -> str:
    if register is None:
        return GENERIC_SYSTEM
    if register not in REGISTERS:
        raise KeyError(f"Unknown register {register!r}. Known: {sorted(REGISTERS)}")
    return REGISTERS[register].style_hint + " No preamble or meta-commentary."


@dataclass
class GenParams:
    max_new_tokens: int = 1500
    temperature: float = 0.85
    top_p: float = 0.9
    repetition_penalty: float = 1.15
    do_sample: bool = True
    seed: int | None = None


class GenFicGenerator:
    def __init__(
        self,
        adapter_path: str | Path,
        base_model: str = DEFAULT_BASE,
        register: str | None = None,
        device_map: str = "auto",
    ):
        self.adapter_path = str(adapter_path)
        self.base_model = base_model
        self.register = register

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=bnb_config,
            device_map=device_map,
            torch_dtype=torch.bfloat16,
        )
        self.model = PeftModel.from_pretrained(base, self.adapter_path)
        self.model.eval()

    @torch.no_grad()
    def generate(
        self,
        brief: str,
        system: str | None = None,
        params: GenParams | None = None,
        **kwargs,
    ) -> str:
        params = params or GenParams(**kwargs)
        sys_prompt = system if system is not None else _system_for_register(self.register)
        prompt = f"[INST] {sys_prompt}\n\n{brief}\n\nBegin the scene now. [/INST]"

        if params.seed is not None:
            torch.manual_seed(params.seed)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        out = self.model.generate(
            **inputs,
            max_new_tokens=params.max_new_tokens,
            temperature=params.temperature,
            top_p=params.top_p,
            repetition_penalty=params.repetition_penalty,
            do_sample=params.do_sample,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        full = self.tokenizer.decode(out[0], skip_special_tokens=True)
        if "[/INST]" in full:
            full = full.split("[/INST]", 1)[-1].strip()
        return full
