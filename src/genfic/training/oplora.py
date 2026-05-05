"""OPLoRA: Orthogonal-Projection LoRA.

Reference: Tushar Jaju et al., IEEE 2026, "Preventing Catastrophic Forgetting in
Cross-Modal Summarization", §III-B (eq. 2):

    Delta_W' = Delta_W - U_k (U_k^T Delta_W)

where:
  Delta_W = B @ A   (the standard LoRA update)
  U_k     = top-k left singular vectors of the dequantized base weight W
  Delta_W' = the projection-corrected LoRA update added to W at forward time

Effect: LoRA updates are constrained to lie in the orthogonal complement of the
top-k singular subspace of the base weight. This preserves the directions in W
that carry the most pre-training information (instruction-following, factual
recall), so the LoRA can learn style without overwriting them.

Why this matters for GenFic
---------------------------
We train one register adapter per literary cluster on top of Mistral-7B-Instruct.
The base model's instruction-following capability — the ability to obey a brief
inside `[INST] ... [/INST]` — is what makes the adapter usable: the user passes
a brief, the adapter shifts the register, the base shape obeys the constraints.
Without OPLoRA, the register adapter tends to flatten that base capability and
the model stops following the brief. With OPLoRA, prompt templates keep working
after fine-tuning.

Usage
-----
After loading a Mistral model with peft.LoraConfig applied:

    from genfic.training.oplora import wrap_with_oplora

    model = wrap_with_oplora(
        model,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        k=16,
    )

Then call `model.reproject_lora_()` after each optimizer step (the trainer
script does this in its hook). Cheap — it's one matmul per target module.
"""

from __future__ import annotations

import torch
import torch.nn as nn

try:
    import bitsandbytes as bnb  # type: ignore
    _HAS_BNB = True
except ImportError:
    _HAS_BNB = False


def _dequantize_weight(linear: nn.Module) -> torch.Tensor:
    """Return a fp32 dense weight matrix [out, in] from any peft-wrapped layer.

    Handles regular nn.Linear and bitsandbytes 4-bit quantized layers
    (`bnb.nn.Linear4bit`). Returns CPU tensor for SVD (which is more numerically
    stable on CPU and avoids GPU OOM during init).
    """
    if _HAS_BNB and isinstance(linear, bnb.nn.Linear4bit):
        # base linear's quantized weight; dequantize via bnb
        w = bnb.functional.dequantize_4bit(
            linear.weight.data, quant_state=linear.weight.quant_state
        )
    else:
        w = linear.weight.data
    return w.detach().to(torch.float32).cpu()


def _compute_top_k_left_singular(weight: torch.Tensor, k: int) -> torch.Tensor:
    """Top-k left singular vectors of `weight` (shape [out, in]) -> shape [out, k]."""
    # Use truncated SVD via torch.svd_lowrank for speed; falls back to full SVD if k is large.
    out_dim = weight.shape[0]
    k_eff = min(k, out_dim - 1)
    if k_eff <= 0:
        return torch.zeros(out_dim, 0, dtype=weight.dtype)
    # svd_lowrank gives U, S, V with U shape [out, q]. q must be > k.
    U, _, _ = torch.svd_lowrank(weight, q=min(k_eff + 8, min(weight.shape)), niter=4)
    return U[:, :k_eff].contiguous()


class OPLoRAReprojector:
    """Holds the cached U_k buffers per target module and reprojects LoRA weights.

    The peft library stores LoRA matrices on each LoraLayer as `lora_A[adapter]`
    (shape [r, in]) and `lora_B[adapter]` (shape [out, r]). Their product BA is
    the additive Delta_W. We apply the projection to BA, then redistribute the
    projected delta back into B (keeping A fixed) via:

        B_new = (Delta_W' @ A^T) @ (A A^T)^{-1}

    where (A A^T)^{-1} is the rank-r inverse of a small r×r Gram matrix —
    cheap. This preserves dimensionality and keeps the optimizer state
    consistent (we only modify .data; gradients are intact).

    Call .reproject_(model) after each `optimizer.step()`.
    """

    def __init__(
        self,
        target_module_names: list[str],
        k: int = 16,
        adapter_name: str = "default",
    ):
        self.target_module_names = set(target_module_names)
        self.k = k
        self.adapter_name = adapter_name
        self._U_k: dict[str, torch.Tensor] = {}  # module qualified name -> U_k tensor

    @torch.no_grad()
    def initialize(self, model: nn.Module) -> None:
        """One-shot SVD pass over all LoRA-wrapped target modules.

        Walks the model, finds `peft.tuners.lora.layer.LoraLayer` instances whose
        leaf-name is in `target_module_names`, dequantizes the base weight, runs
        truncated SVD, and stores U_k as a CPU tensor (small per layer).
        """
        try:
            from peft.tuners.lora.layer import LoraLayer
        except ImportError as e:
            raise RuntimeError(
                "OPLoRA requires peft. Install with `pip install peft`."
            ) from e

        n_init = 0
        for qname, module in model.named_modules():
            if not isinstance(module, LoraLayer):
                continue
            leaf = qname.split(".")[-1]
            if leaf not in self.target_module_names:
                continue
            base = module.base_layer if hasattr(module, "base_layer") else module
            w = _dequantize_weight(base)
            U_k = _compute_top_k_left_singular(w, self.k)
            self._U_k[qname] = U_k
            n_init += 1
        if n_init == 0:
            raise RuntimeError(
                f"OPLoRA found no target LoraLayers; got names {self.target_module_names}"
            )
        # Sanity log
        print(f"[OPLoRA] initialized U_k on {n_init} target modules (k={self.k})")

    @torch.no_grad()
    def reproject_(self, model: nn.Module) -> None:
        """Reproject every cached LoRA delta to be orthogonal to its U_k.

        Modifies B.data in-place. A is unchanged. Optimizer state remains valid
        since we only adjusted parameter values, not their shapes.
        """
        try:
            from peft.tuners.lora.layer import LoraLayer
        except ImportError:
            return  # init would have raised

        for qname, module in model.named_modules():
            if not isinstance(module, LoraLayer) or qname not in self._U_k:
                continue
            U_k = self._U_k[qname]
            if U_k.numel() == 0:
                continue

            A = module.lora_A[self.adapter_name].weight.data  # [r, in]
            B = module.lora_B[self.adapter_name].weight.data  # [out, r]
            device, dtype = B.device, B.dtype
            U_k_dev = U_k.to(device=device, dtype=dtype)

            # Delta W = B @ A  (shape [out, in])
            delta = B @ A
            # Project orthogonal to U_k:  delta' = delta - U_k (U_k^T delta)
            delta_proj = delta - U_k_dev @ (U_k_dev.t() @ delta)

            # Redistribute into B: solve B_new @ A = delta_proj, with A fixed.
            #   B_new = (delta_proj @ A^T) @ (A @ A^T)^{-1}
            # AAT is [r, r] — small (r=32 typical), cheap.
            AAT = A @ A.t()
            # Add tiny diagonal to avoid singular Gram if A is rank-deficient at init.
            r = AAT.shape[0]
            AAT = AAT + torch.eye(r, device=device, dtype=dtype) * 1e-6
            B_new = (delta_proj @ A.t()) @ torch.linalg.inv(AAT)
            B.copy_(B_new)


def wrap_with_oplora(
    model: nn.Module,
    target_modules: list[str],
    k: int = 16,
    adapter_name: str = "default",
) -> tuple[nn.Module, OPLoRAReprojector]:
    """Convenience helper: build the reprojector, run init, return both.

    Caller must call `reprojector.reproject_(model)` after each optimizer step.
    """
    rep = OPLoRAReprojector(
        target_module_names=target_modules, k=k, adapter_name=adapter_name
    )
    rep.initialize(model)
    return model, rep
