from dataclasses import asdict, dataclass, field
import math

from weft.meta import ESTIMATE
from weft.vision import TrackingConfig


@dataclass(frozen=True)
class RunConfig:
    episodes: int = 15000
    eval_episodes: int = 300
    seed: int = 1729
    eval_seed: int = 938401
    provider: str = "local"
    repair_enabled: bool = True
    meta_budget: float | None = None
    tracking: TrackingConfig = field(default_factory=TrackingConfig)

    def validate(self):
        for name, maximum in (("episodes", 50000), ("eval_episodes", 2000)):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= maximum:
                raise ValueError(f"{name} must be an integer between 1 and {maximum}")
        for name in ("seed", "eval_seed"):
            value = getattr(self, name)
            if type(value) is not int or not 0 <= value < 2 ** 32 - 1:
                raise ValueError(f"{name} must be an integer in [0, 2**32 - 1)")
        if self.seed == self.eval_seed:
            raise ValueError("Training and evaluation seeds must differ")
        if self.provider not in ("local", "meta") or type(self.repair_enabled) is not bool:
            raise ValueError("Invalid provider or repair_enabled setting")
        if self.provider == "meta":
            if (not self.repair_enabled or type(self.meta_budget) not in (int, float)
                    or not math.isfinite(self.meta_budget) or self.meta_budget < ESTIMATE):
                raise ValueError("Meta requires repairs enabled and a finite explicit budget covering its estimate")
        elif self.meta_budget is not None:
            raise ValueError("A Meta budget is only valid with provider=meta")
        self.tracking.validate()
        return self

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or set(value) - set(cls.__dataclass_fields__):
            raise ValueError("Run config contains unknown fields or is not an object")
        data = dict(value)
        tracking = data.pop("tracking", {})
        if not isinstance(tracking, dict) or set(tracking) - set(TrackingConfig.__dataclass_fields__):
            raise ValueError("Tracking config contains unknown fields or is not an object")
        return cls(**data, tracking=TrackingConfig(**tracking)).validate()
