"""Train a literary-register OPLoRA adapter on Mistral-7B-Instruct-v0.2.

Hyperparameters lifted from the IEEE paper §IV-B; LoRA+ asymmetric LRs from
Hayou et al. 2024. Fits within ~12 GB VRAM with grad-checkpointing on RTX 5070 Ti.

Usage
-----
  # Train the victorian-formal adapter with defaults
  python scripts/train.py --register victorian-formal

  # Quick smoke
  python scripts/train.py --register victorian-formal --max-steps 50 --eval-steps 25

  # Ablation against vanilla LoRA (no orthogonal projection)
  python scripts/train.py --register victorian-formal --no-oplora --out runs/victorian-formal-no-oplora

Outputs
-------
  runs/{register}/
    checkpoint-XXX/              periodic LoRA snapshots
    final/                       final adapter + tokenizer
    training_invocation.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch  # noqa: E402
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training  # noqa: E402
from transformers import (  # noqa: E402
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from genfic.registers import REGISTERS  # noqa: E402
from genfic.training.dataset import GenFicDataset  # noqa: E402
from genfic.training.oplora import OPLoRAReprojector  # noqa: E402

TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


class OPLoRACallback(TrainerCallback):
    """Reproject LoRA weights after every optimizer step."""

    def __init__(self, reprojector: OPLoRAReprojector, model):
        self.reprojector = reprojector
        self.model = model

    def on_step_end(self, args, state, control, **kwargs):
        self.reprojector.reproject_(self.model)


def build_optimizer_with_lora_plus(
    model, lr_a: float, lr_b: float, weight_decay: float
):
    """LoRA+ asymmetric learning rates: lora_B trained η× faster than lora_A.
    Per IEEE paper §IV-B and Hayou et al. 2024, η=8 typical.
    """
    a_params, b_params, other_params = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "lora_A" in name:
            a_params.append(p)
        elif "lora_B" in name:
            b_params.append(p)
        else:
            other_params.append(p)
    groups = [
        {"params": a_params, "lr": lr_a, "weight_decay": weight_decay},
        {"params": b_params, "lr": lr_b, "weight_decay": weight_decay},
    ]
    if other_params:
        groups.append({"params": other_params, "lr": lr_a, "weight_decay": weight_decay})
    try:
        import bitsandbytes as bnb
        return bnb.optim.AdamW8bit(groups, betas=(0.9, 0.999), eps=1e-8)
    except ImportError:
        from torch.optim import AdamW
        return AdamW(groups, betas=(0.9, 0.999), eps=1e-8)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--register", required=True, choices=sorted(REGISTERS),
                   help="Which register cluster's dataset to train on")
    p.add_argument("--model", default="mistralai/Mistral-7B-Instruct-v0.2")
    p.add_argument("--data", default=None,
                   help="Override dataset path (default: source/{register}.h5)")
    p.add_argument("--out", default=None,
                   help="Override output dir (default: runs/{register})")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--max-steps", type=int, default=-1,
                   help="Cap total training steps (overrides epochs if > 0)")
    p.add_argument("--micro-batch", type=int, default=3)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr-a", type=float, default=1e-4)
    p.add_argument("--lr-b", type=float, default=8e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--lora-r", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=64)
    p.add_argument("--lora-dropout", type=float, default=0.1)
    p.add_argument("--oplora-k", type=int, default=64)
    p.add_argument("--no-oplora", action="store_true",
                   help="Disable OPLoRA (ablation A/B against vanilla LoRA)")
    p.add_argument("--save-steps", type=int, default=250)
    p.add_argument("--eval-steps", type=int, default=250)
    p.add_argument("--logging-steps", type=int, default=20)
    p.add_argument("--early-stop-patience", type=int, default=2,
                   help="Stop after N consecutive evals with improvement < threshold")
    p.add_argument("--early-stop-threshold", type=float, default=0.02,
                   help="Minimum eval_loss improvement to reset patience")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if args.save_steps % args.eval_steps != 0:
        print(f"ERROR: --save-steps ({args.save_steps}) must be a multiple of "
              f"--eval-steps ({args.eval_steps}) so load_best_model_at_end works.",
              file=sys.stderr)
        return 2

    out_dir = REPO_ROOT / (args.out or f"runs/{args.register}")
    data_path = REPO_ROOT / (args.data or f"source/{args.register}.h5")
    out_dir.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        print(f"ERROR: dataset not found at {data_path}. "
              f"Run scripts/build_dataset.py --register {args.register} first.",
              file=sys.stderr)
        return 2

    print(f"Loading tokenizer + model: {args.model}")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    print(f"  base loaded in {time.time()-t0:.0f}s")

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.config.use_cache = False

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=TARGET_MODULES,
    )
    model = get_peft_model(model, lora_cfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    reprojector = None
    if not args.no_oplora:
        t0 = time.time()
        reprojector = OPLoRAReprojector(
            target_module_names=TARGET_MODULES, k=args.oplora_k
        )
        reprojector.initialize(model)
        print(f"  OPLoRA SVD setup: {time.time()-t0:.0f}s")

    train_ds = GenFicDataset(data_path, "train")
    val_ds = GenFicDataset(data_path, "val")
    print(f"  register={train_ds.register}  train chunks: {len(train_ds):,}  val chunks: {len(val_ds):,}")

    targs = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=args.micro_batch,
        per_device_eval_batch_size=args.micro_batch,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr_a,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=[],
        seed=args.seed,
        dataloader_num_workers=2,
        remove_unused_columns=False,
    )

    optimizer = build_optimizer_with_lora_plus(
        model, args.lr_a, args.lr_b, args.weight_decay
    )

    callbacks = [
        EarlyStoppingCallback(
            early_stopping_patience=args.early_stop_patience,
            early_stopping_threshold=args.early_stop_threshold,
        ),
    ]
    if reprojector is not None:
        callbacks.append(OPLoRACallback(reprojector, model))

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        optimizers=(optimizer, None),
        callbacks=callbacks,
    )

    (out_dir / "training_invocation.json").write_text(
        json.dumps(vars(args), indent=2), encoding="utf-8"
    )

    print(f"\nStarting training: register={args.register}, "
          f"epochs={args.epochs}, micro_batch={args.micro_batch} x grad_accum={args.grad_accum}")
    print(f"  effective batch = {args.micro_batch * args.grad_accum}")
    print(f"  LRs: lora_A={args.lr_a}, lora_B={args.lr_b} (eta={args.lr_b/args.lr_a:.0f}x)")
    print(f"  OPLoRA: {'OFF' if args.no_oplora else f'k={args.oplora_k}'}")
    print(f"  EarlyStopping: patience={args.early_stop_patience}, "
          f"threshold={args.early_stop_threshold} on eval_loss")
    trainer.train()

    print("\nSaving final adapter...")
    trainer.save_model(str(out_dir / "final"))
    tokenizer.save_pretrained(str(out_dir / "final"))
    print(f"Adapter -> {out_dir / 'final'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
