import re
import warnings
from itertools import chain

import torch
from peft.tuners.lora import LoraModel
from peft.tuners.tuners_utils import BaseTunerLayer
from peft.utils import _freeze_adapter
from transformers.pytorch_utils import Conv1D

from .layer import L1RALayer, L1RALinear


class L1RAModel(LoraModel):
    """LoRA с обучаемым gate-вектором для каждой ранговой компоненты."""

    def __init__(self, model, config, adapter_name):
        super().__init__(model, config, adapter_name)

        trainable = sum(1 for c in self.peft_config.values() if not c.inference_mode)
        if trainable > 1:
            raise ValueError("L1RAModel supports only 1 trainable adapter at a time.")

        if config.inference_mode:
            _freeze_adapter(self.model, adapter_name)
        else:
            self.trainable_adapter_name = adapter_name

    def _create_and_replace(self, lora_config, adapter_name, target, target_name, parent, current_key):
        pattern_keys = list(chain(lora_config.rank_pattern.keys(), lora_config.alpha_pattern.keys()))
        target_name_key = next(
            filter(lambda k: re.match(rf".*\.{k}$", current_key), pattern_keys), current_key
        )
        r     = lora_config.rank_pattern.get(target_name_key, lora_config.r)
        alpha = lora_config.alpha_pattern.get(target_name_key, lora_config.lora_alpha)

        kwargs = dict(
            r=r,
            lora_alpha=alpha,
            lora_dropout=lora_config.lora_dropout,
            fan_in_fan_out=lora_config.fan_in_fan_out,
            init_lora_weights=lora_config.init_lora_weights,
        )

        if not isinstance(target, L1RALayer):
            new_module = self._create_new_module(lora_config, adapter_name, target, **kwargs)
            if adapter_name != self.active_adapter:
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)
        else:
            target.update_layer(adapter_name, r, alpha, lora_config.lora_dropout, lora_config.init_lora_weights)

    @staticmethod
    def _create_new_module(lora_config, adapter_name, target, **kwargs):
        base = target.get_base_layer() if isinstance(target, BaseTunerLayer) else target

        if isinstance(base, torch.nn.Linear):
            if kwargs.get("fan_in_fan_out"):
                warnings.warn("fan_in_fan_out=True для nn.Linear; устанавливаю False.")
                kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = False
        elif isinstance(base, Conv1D):
            if not kwargs.get("fan_in_fan_out"):
                warnings.warn("fan_in_fan_out=False для Conv1D; устанавливаю True.")
                kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = True
        else:
            raise ValueError(f"Unsupported target module type: {type(base)}")

        return L1RALinear(target, adapter_name, **kwargs)

    def print_trainable_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        pct = 100 * trainable / total if total > 0 else 0
        print(
            f"trainable params: {trainable:,} || "
            f"all params: {total:,} || "
            f"trainable%: {pct:.4f}"
        )

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)
