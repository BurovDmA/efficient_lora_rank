import torch
from transformers import AutoModelForCausalLM
from peft import LoraConfig, AdaLoraConfig, get_peft_model

from .l1ra import L1RAConfig, L1RAModel


def _load_base_model(model_name: str):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.config.use_cache = False
    return model


def build_lora_model(rank: int, model_name: str):
    model = _load_base_model(model_name)

    config = LoraConfig(
        r=rank,
        lora_alpha=rank * 2,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )

    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model


def build_adalora_model(
    init_r: int,
    target_r: int,
    model_name: str,
    total_step: int = None,
    tinit: int = None,
    tfinal: int = None,
    deltaT: int = 10,
):
    """AdaLoRA с заданным стартовым и целевым рангом."""
    if total_step is None or total_step <= 0:
        raise ValueError(
            "`total_step` must be a positive integer. "
            "Pass `total_step = (len(train_loader) // grad_accum) * num_epochs`."
        )

    tinit  = tinit  if tinit  is not None else max(1, int(total_step * 0.10))
    tfinal = tfinal if tfinal is not None else max(tinit + 1, int(total_step * 0.70))

    model = _load_base_model(model_name)

    config = AdaLoraConfig(
        init_r=init_r,
        target_r=target_r,
        lora_alpha=init_r * 2,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
        total_step=total_step,
        tinit=tinit,
        tfinal=tfinal,
        deltaT=deltaT,
    )

    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model


def build_l1ra_model(
    rank: int,
    model_name: str,
    l1ra_lambda: float = 1e-3,
    eta_c: float = 1e-2,
):
    """L1RA: LoRA с L1-регуляризацией gate-векторов."""
    base_model = _load_base_model(model_name)

    config = L1RAConfig(
        r=rank,
        lora_alpha=rank * 2,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
        l1ra_lambda=l1ra_lambda,
        eta_c=eta_c,
    )

    model = L1RAModel(base_model, config, "default")
    model.print_trainable_parameters()
    return model
