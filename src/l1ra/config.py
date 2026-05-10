from dataclasses import dataclass, field

from peft.tuners.lora import LoraConfig


@dataclass
class L1RAConfig(LoraConfig):
    """Конфиг L1RA."""

    l1ra_lambda: float = field(
        default=1e-3, metadata={"help": "Коэффициент L1-регуляризации gate-векторов."}
    )
    eta_c: float = field(
        default=1e-2, metadata={"help": "Отдельный learning rate для gate-векторов."}
    )
