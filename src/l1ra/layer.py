import warnings
from typing import Any, List, Optional

import torch
from peft.tuners.lora import LoraLayer
from peft.tuners.tuners_utils import check_adapters_to_merge
from peft.utils import transpose
from torch import nn


class L1RALayer(LoraLayer):
    adapter_layer_names = ("lora_A", "lora_B", "lora_c", "lora_embedding_A", "lora_embedding_B")

    def __init__(self, base_layer: nn.Module, **kwargs) -> None:
        super().__init__(base_layer)
        self.lora_c = nn.ParameterDict({})
        self.lora_A = nn.ParameterDict({})
        self.lora_B = nn.ParameterDict({})

    def update_layer(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights):
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer, but got {r}")

        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha

        lora_dropout_layer = nn.Dropout(p=lora_dropout) if lora_dropout > 0.0 else nn.Identity()
        self.lora_dropout[adapter_name] = lora_dropout_layer

        self.lora_A[adapter_name] = nn.Parameter(torch.zeros(self.in_features, r))
        self.lora_B[adapter_name] = nn.Parameter(torch.zeros(r, self.out_features))
        self.lora_c[adapter_name] = nn.Parameter(torch.ones(r))
        self.scaling[adapter_name] = lora_alpha / r

        if init_lora_weights:
            self.reset_lora_parameters(adapter_name)

        base = self.get_base_layer()
        device = base.qweight.device if hasattr(base, "qweight") else base.weight.device
        self.to(device)
        self.set_adapter(self.active_adapters)

    def reset_lora_parameters(self, adapter_name):
        if adapter_name in self.lora_A:
            nn.init.normal_(self.lora_A[adapter_name], std=1 / self.r[adapter_name])
            nn.init.zeros_(self.lora_B[adapter_name])
            nn.init.ones_(self.lora_c[adapter_name])


class L1RALinear(nn.Module, L1RALayer):
    def __init__(
        self,
        base_layer: nn.Module,
        adapter_name: str,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,
        init_lora_weights: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        L1RALayer.__init__(self, base_layer, **kwargs)
        self.get_base_layer().weight.requires_grad = False
        self.fan_in_fan_out = fan_in_fan_out
        self._active_adapter = adapter_name
        self.update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights)

    def merge(self, safe_merge: bool = False, adapter_names: Optional[List[str]] = None) -> None:
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            return
        for active_adapter in adapter_names:
            if active_adapter in self.lora_A:
                base_layer = self.get_base_layer()
                if safe_merge:
                    orig = base_layer.weight.data.clone()
                    orig += self.get_delta_weight(active_adapter)
                    if not torch.isfinite(orig).all():
                        raise ValueError(f"NaNs in merged weights for adapter {active_adapter}")
                    base_layer.weight.data = orig
                else:
                    base_layer.weight.data += self.get_delta_weight(active_adapter)
                self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        while self.merged_adapters:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.lora_A:
                self.get_base_layer().weight.data -= self.get_delta_weight(active_adapter)

    def get_delta_weight(self, adapter) -> torch.Tensor:
        delta = (self.lora_A[adapter] * self.lora_c[adapter]) @ self.lora_B[adapter]
        return transpose(delta.T, self.fan_in_fan_out) * self.scaling[adapter]

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            return self.base_layer(x, *args, **kwargs)

        if self.merged:
            return self.base_layer(x, *args, **kwargs)

        result = self.base_layer(x, *args, **kwargs)
        torch_result_dtype = result.dtype

        for active_adapter in self.active_adapters:
            if active_adapter not in self.lora_A:
                continue
            lora_A   = self.lora_A[active_adapter]
            lora_B   = self.lora_B[active_adapter]
            lora_c   = self.lora_c[active_adapter]
            dropout  = self.lora_dropout[active_adapter]
            scaling  = self.scaling[active_adapter]

            lora_c.data.clamp_(0.0, 1.0)
            x_adapter = dropout(x).to(lora_A.dtype)
            delta = ((x_adapter @ lora_A) * lora_c) @ lora_B
            result = result + (delta * scaling).to(torch_result_dtype)

        return result.to(torch_result_dtype)
