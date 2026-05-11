"""CLI for one-off generation runs against a trained register adapter.

Examples
--------
  # By register name (resolves adapter path automatically)
  python scripts/generate.py --register victorian-formal \\
      --brief "An afternoon visit to the parsonage; the vicar's wife is unwell."

  # Explicit adapter path (overrides --register's path resolution but the
  # register's style hint still drives the system prompt)
  python scripts/generate.py --register gothic-dark \\
      --adapter runs/gothic-dark/checkpoint-750 \\
      --brief-file briefs/abandoned-chapel.txt --max-tokens 2000

  # Base model only (no adapter) — sanity-check baseline
  python scripts/generate.py --register plain-realist --no-adapter \\
      --brief "A child watches a snowstorm through a kitchen window."
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from genfic.inference.generate import GenFicGenerator, GenParams  # noqa: E402
from genfic.registers import REGISTERS  # noqa: E402


def resolve_adapter(register: str, explicit: str | None) -> Path | None:
    if explicit:
        return Path(explicit)
    run_dir = REPO_ROOT / "runs" / register
    if not run_dir.exists():
        return None
    # Prefer "final" if it exists, else newest checkpoint-*
    final = run_dir / "final"
    if final.exists():
        return final
    ckpts = sorted(
        (p for p in run_dir.iterdir() if p.is_dir() and p.name.startswith("checkpoint-")),
        key=lambda p: int(p.name.split("-")[-1]),
    )
    return ckpts[-1] if ckpts else None


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--register", required=True, choices=sorted(REGISTERS),
                   help="Which register cluster's adapter and style hint to use")
    p.add_argument("--adapter", help="Override adapter path (default: runs/{register}/final or newest checkpoint)")
    p.add_argument("--no-adapter", action="store_true",
                   help="Skip adapter; run base model with the register's system prompt only")
    p.add_argument("--base", default="mistralai/Mistral-7B-Instruct-v0.2")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--brief", help="Inline brief text")
    g.add_argument("--brief-file", help="File containing the brief")
    p.add_argument("--system", default=None,
                   help="Override the system message (only used with --no-adapter; "
                        "adapter mode uses the training-format prompt for parity)")
    p.add_argument("--max-tokens", type=int, default=1500)
    p.add_argument("--temperature", type=float, default=0.85)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--repetition-penalty", type=float, default=1.15)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--out", default=None, help="Optional file to write the output to")
    args = p.parse_args()

    brief = args.brief or Path(args.brief_file).read_text(encoding="utf-8")

    if args.no_adapter:
        # Lightweight base-only path: we still want a register-aware system prompt.
        # Build a minimal generator with no adapter.
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        import torch
        bnb = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
        )
        tok = AutoTokenizer.from_pretrained(args.base)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            args.base, quantization_config=bnb,
            device_map="auto", torch_dtype=torch.bfloat16,
        )
        model.eval()
        from genfic.inference.generate import _baseline_system
        sys_prompt = args.system or _baseline_system(args.register)
        prompt = f"[INST] {sys_prompt}\n\n{brief}\n\nBegin the scene now. [/INST]"
        if args.seed is not None:
            torch.manual_seed(args.seed)
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=args.max_tokens,
                temperature=args.temperature, top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                do_sample=True, pad_token_id=tok.eos_token_id,
            )
        full = tok.decode(out[0], skip_special_tokens=True)
        text = full.split("[/INST]", 1)[-1].strip() if "[/INST]" in full else full
    else:
        adapter = resolve_adapter(args.register, args.adapter)
        if adapter is None or not adapter.exists():
            print(f"ERROR: no adapter found for --register {args.register}. "
                  f"Pass --adapter PATH or --no-adapter.", file=sys.stderr)
            return 2
        print(f"Loading adapter: {adapter}")
        if args.system is not None:
            print("WARNING: --system is ignored in adapter mode (training-format "
                  "parity). Use --no-adapter if you want to override the system prompt.",
                  file=sys.stderr)
        gen = GenFicGenerator(
            adapter_path=adapter, base_model=args.base, register=args.register,
        )
        print(f"Generating ({args.max_tokens} max tokens, temp={args.temperature}) ...")
        text = gen.generate(
            brief,
            params=GenParams(
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                seed=args.seed,
            ),
        )

    print("\n" + "=" * 60 + "\n")
    print(text)
    print("\n" + "=" * 60)
    print(f"Output: {len(text):,} chars / ~{len(text.split()):,} words")
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"Saved to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
